"""Activation-space regularisation on the residual stream.

Why activation-space (not weight-space)
---------------------------------------
σReparam, rank-floor, weight-decay are all *weight*-space tools — they
constrain the matrix shape but say nothing about what the network
actually computes on a given input. The K7 / T0 / T11 forensic
threads showed that *runtime* failure modes (V-norm explosion, image
v_norm blowing up to >50, the residual stream sliding off the
LayerNorm sweet spot) often *precede* the weight-shape pathology by
several thousand steps. Acting directly on the activations gives us a
low-overhead, behaviour-targeted lever.

Two modes are supported:

* ``norm_target`` (default) — penalises ``(‖x_l‖_2 / √D − τ)²`` per
  decoder-layer output ``x_l``. The square-root-of-D normalisation
  matches the "RMS norm" used by Llama / DINOv3 / Pythia (where every
  healthy hidden state has per-token RMS ≈ 1). With ``τ=1`` this
  simply asks the residual stream not to drift away from the
  unit-RMS regime.
* ``inter_layer_lipschitz`` — penalises
  ``(‖x_l − x_{l−1}‖_2² / ‖x_{l−1}‖_2² − τ)`` clamped at zero. The
  ratio is the per-layer "step size" of the residual stream; capping
  it stops a single sub-block from dominating the forward pass (the
  pathology behind T11's gate-compounding analysis).

Hook lifecycle
--------------
The regulariser registers forward hooks at construction and **stays
registered for the lifetime of the model**. Each hook appends the
output tensor to an internal buffer when the regulariser is *enabled*,
otherwise the hook is an early-return — the cost in eval / inference
is one ``if not self._enabled: return`` per layer, ~zero.

Use as a context manager *or* via explicit ``start()/stop()``:

.. code-block:: python

    reg = ActivationRegularizer(model, mode="norm_target", weight=1e-2)
    # ...
    def training_step(self, batch, batch_idx):
        with reg:
            pred = self.model(...)
            main_loss = self._compute_loss(pred, gt)
            return main_loss + reg.compute_loss()

The buffer is cleared every time we enter the context, so each
training step gets a fresh capture.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


VALID_MODES = ("norm_target", "inter_layer_lipschitz")


class ActivationRegularizer:
    """Forward-hook-based activation regulariser.

    Construction installs forward hooks on every module in the model
    whose qualified name matches ``hook_filter``. Default filter
    captures every decoder layer's final output (the residual stream
    after ``FFN`` sub-block); for V3 we additionally capture the V3
    decoder's posterior-update output so we can observe the bank's
    drift across the 10 shared updater calls.

    Args:
        model: the full Lightning-wrapped model. Hooks are attached to
            submodules of ``model`` (i.e. usually ``pl_module.model``).
        mode: ``"norm_target"`` or ``"inter_layer_lipschitz"``.
        weight: regularisation strength. ``0`` ⇒ no-op (still records
            activations for diagnostics).
        norm_target: ``τ`` for ``norm_target`` mode. Default 1.0
            (per-token RMS = 1 ≈ DINOv3 / Llama default).
        lipschitz_threshold: ``τ`` for ``inter_layer_lipschitz``.
            Default 0.5 (each layer adds ≤ √0.5·prev RMS).
        hook_filter: regex applied to ``module.named_modules()`` keys.
            Default targets V2 ``HeightDecoderStack.layers.*.ffn`` and
            V3 ``HeightDecoderStackV3.layers.*.ffn`` plus the V3
            posterior, since those are the canonical residual-stream
            checkpoints in our architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str = "norm_target",
        weight: float = 1e-2,
        norm_target: float = 1.0,
        lipschitz_threshold: float = 0.5,
        hook_filter: str = r"\.layers\.\d+\.ffn$",
    ):
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}, got {mode!r}"
            )
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")

        self.mode = mode
        self.weight = float(weight)
        self.norm_target = float(norm_target)
        self.lipschitz_threshold = float(lipschitz_threshold)

        self._enabled = False
        self._buffers: List[torch.Tensor] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._hook_filter = re.compile(hook_filter)

        targets = self._discover_targets(model)
        for name, module in targets:
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

        logger.info(
            f"ActivationRegularizer: tracking {len(self._handles)} sites "
            f"(mode={mode!r}, weight={self.weight:g}, "
            f"filter={hook_filter!r})"
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------
    def _discover_targets(
        self, model: nn.Module
    ) -> List[Tuple[str, nn.Module]]:
        targets: List[Tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if self._hook_filter.search(name):
                targets.append((name, module))
        # Stable order: sort by qualified name so 'layers.0' < 'layers.10'
        # logically — pad numeric suffixes when sorting.
        def _sort_key(item):
            n = item[0]
            return tuple(int(p) if p.isdigit() else p
                         for p in re.split(r"(\d+)", n))
        targets.sort(key=_sort_key)
        return targets

    def _make_hook(self, _name: str):
        def _hook(_module, _inputs, output):
            if not self._enabled:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                self._buffers.append(tensor)
        return _hook

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Context-manager / start-stop API
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._buffers.clear()
        self._enabled = True

    def stop(self) -> None:
        self._enabled = False

    def __enter__(self) -> "ActivationRegularizer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self) -> torch.Tensor:
        if not self._buffers:
            return torch.zeros((), dtype=torch.float32)
        device = self._buffers[0].device
        if self.weight == 0.0:
            return torch.zeros((), device=device, dtype=torch.float32)

        if self.mode == "norm_target":
            losses = []
            for x in self._buffers:
                # x: [B, P, D] (residual stream tokens)
                D = x.shape[-1]
                norm_per_token = x.float().norm(dim=-1)  # [B, P]
                rms_per_token = norm_per_token / (D ** 0.5)
                losses.append(((rms_per_token - self.norm_target) ** 2).mean())
            return self.weight * torch.stack(losses).mean()

        # mode == "inter_layer_lipschitz"
        if len(self._buffers) < 2:
            return torch.zeros((), device=device, dtype=torch.float32)
        losses = []
        for i in range(1, len(self._buffers)):
            prev = self._buffers[i - 1].float()
            cur = self._buffers[i].float()
            delta_sq = ((cur - prev) ** 2).sum(dim=-1)         # [B, P]
            prev_sq = (prev ** 2).sum(dim=-1).clamp(min=1e-8)  # [B, P]
            ratio = delta_sq / prev_sq
            shortfall = (ratio - self.lipschitz_threshold).clamp(min=0.0)
            losses.append(shortfall.mean())
        return self.weight * torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def num_hooks(self) -> int:
        return len(self._handles)

    @torch.no_grad()
    def last_norms(self) -> List[float]:
        """Per-site mean RMS over the last captured forward. Empty if
        the regulariser hasn't seen a forward pass yet."""
        out: List[float] = []
        for x in self._buffers:
            D = x.shape[-1]
            rms = x.float().norm(dim=-1).mean() / (D ** 0.5)
            out.append(float(rms.item()))
        return out
