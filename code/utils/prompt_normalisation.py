"""Per-sample affine renormalisation of (prompt, target) by prompt extents.

Background
----------
PromptDA (§3.4 "Depth normalization") computes per-sample
``(scale, shift) = (L_max - L_min, L_min)`` from the LiDAR prompt and
trains the network to predict ``(D_target - shift) / scale`` so the
target dynamic range is uniform across samples and metric scale is
delegated to the prompt at inference time.

This module exposes that operation as a small, dataset-agnostic helper
so any height-style dataset (HyperSim, SynRS3D, Open-Canopy, ARKitScenes)
can opt into it via a single config flag without re-implementing the
arithmetic and without coupling to PromptDA's other design choices.

Two semantic differences vs PromptDA worth flagging:

* We compute ``(shift, scale)`` from the **clean** prompt (= GT in
  synthetic-data setups, or the raw uncorrupted prompt for real data),
  *before* :class:`CHMCorruptor` runs. This decouples the normalisation
  factor from the corruption realisation -- a corrupted prompt with
  cutout zeros would otherwise pull ``L_min`` down to 0 and inflate the
  scale by an arbitrary factor that depends on the random crop. The
  user explicitly requested this ordering (min-max BEFORE cutout).
* The validation metric path is responsible for *un*-normalising the
  prediction back to metres before MAE/RMSE/δ<1.25 are computed, so
  metrics stay comparable across runs that disagree on whether
  normalisation is enabled.

Usage
-----
At the dataset boundary::

    prompt_clean, gt = read_clean_pair(...)
    if minmax_normalise:
        from utils.prompt_normalisation import minmax_normalise_pair
        prompt_clean, gt, shift, scale = minmax_normalise_pair(
            prompt_clean, gt, eps=1e-3, min_scale=1e-2,
        )
        meta = np.array([shift, scale], dtype=np.float32)
    prompt = corruptor(prompt_clean)         # corruption runs on normalised values
    return image, prompt, gt, meta           # 4-tuple when norm is enabled

At the lightning module::

    pred_metric = pred_norm * scale + shift
    gt_metric   = gt_norm   * scale + shift
    mae = (pred_metric - gt_metric).abs().mean()
"""

from __future__ import annotations

import numpy as np


def minmax_normalise_pair(
    prompt: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    eps: float = 1.0e-6,
    min_scale: float = 1.0e-2,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Affine-renormalise ``prompt`` and ``target`` by the prompt's min/max.

    The same ``(shift, scale)`` is applied to both arrays so any
    geometric relationship between prompt and target survives
    unchanged in the normalised frame.

    Args:
        prompt:    ``[H, W]`` float array (the clean LiDAR / CHM prompt).
        target:    ``[H, W]`` float array (the regression target).
        valid_mask: optional boolean mask; when provided, only pixels
                    where ``valid_mask`` is True contribute to the
                    min/max. Default: ``prompt > 0`` (treat 0 as the
                    standard sentinel).
        eps:       Floor added to the raw range before clamping
                    (numerical safety -- almost never relevant).
        min_scale: Lower bound on the affine scale. Prevents
                    pathological zero-range prompts (uniform-height
                    scenes, fully zeroed prompt) from producing
                    division-by-zero or astronomical normalised
                    values. ``0.01`` translates to "any scene with
                    less than 1 cm of prompt range is treated as if
                    it had 1 cm of range". Increase if your domain
                    has a meaningful minimum (e.g. set to 1.0 for
                    forest height where < 1 m range = effectively flat).

    Returns:
        ``(prompt_norm, target_norm, shift, scale)`` where
        ``prompt_norm = (prompt - shift) / scale`` is approximately in
        ``[0, 1]`` (exactly so over the valid mask), and
        ``target_norm`` may exceed ``[0, 1]`` if the target spans a
        larger range than the prompt -- this is by design and handled
        downstream by the loss without special-casing.
    """
    if prompt.shape != target.shape:
        raise ValueError(
            f"prompt and target shape mismatch: "
            f"{prompt.shape} vs {target.shape}"
        )

    if valid_mask is None:
        valid_mask = (prompt > 0) & np.isfinite(prompt)
    elif valid_mask.shape != prompt.shape:
        raise ValueError(
            f"valid_mask shape mismatch: "
            f"{valid_mask.shape} vs {prompt.shape}"
        )

    if not valid_mask.any():
        # Degenerate case: no valid prompt pixels (fully dropped /
        # all-sky). Fall back to identity (shift=0, scale=1) so the
        # downstream loss sees the raw values; the encoder will treat
        # the input as its own scale frame.
        return prompt.astype(np.float32), target.astype(np.float32), 0.0, 1.0

    valid_vals = prompt[valid_mask]
    h_min = float(valid_vals.min())
    h_max = float(valid_vals.max())

    raw_scale = max(h_max - h_min, eps)
    scale = max(raw_scale, float(min_scale))

    prompt_norm = ((prompt - h_min) / scale).astype(np.float32)
    target_norm = ((target - h_min) / scale).astype(np.float32)
    return prompt_norm, target_norm, h_min, scale


def invert_minmax(
    value_norm,
    shift,
    scale,
):
    """Recover metric values from a normalised prediction / GT.

    Accepts numpy arrays or torch tensors interchangeably (the operation
    is a single multiply-add). Broadcasting follows numpy / torch
    rules; in the typical training case ``value_norm`` is
    ``[B, 1, H, W]`` and ``shift, scale`` are ``[B]`` or ``[B, 1, 1, 1]``.
    """
    return value_norm * scale + shift
