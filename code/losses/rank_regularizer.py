"""Rank-floor regularisation on the *weight* matrices of the trainable surface.

Why rank-floor regularisation
-----------------------------
The K6 / T1 / T10 / T11 forensic timeline showed *spectral collapse*
of the decoder cross-attention V projection — Frobenius norm stays
flat while the top singular value triples and the entropy-based
effective rank halves. σReparam (Zhai et al. 2023) caps σ_max but
does not bound the spectrum *shape*; once enough mass concentrates in
the top component, the layer's output becomes a rank-1 projection of
its input and downstream attention degenerates regardless of γ.

Rank-floor regularisation adds an explicit penalty on the *ratio*

    stable_rank(W) = ‖W‖_F² / σ_max(W)²

The lower bound is `stable_rank ≥ 1` (rank-1) and the upper bound is
`stable_rank ≤ min(d_out, d_in)` (full-rank uniform spectrum). Healthy
attention layers in foundation transformers run at
`stable_rank ≈ 0.05 · D` to `0.3 · D` — so for our 1024-dim decoder
we target a floor between 64 and 256.

Implementation choice — *act on the parametrised view*
------------------------------------------------------
We compute Frobenius norm on ``linear.weight`` *as the model uses it*
(the parametrised view, post σReparam scaling). That way:

* The optimiser can drive ``stable_rank`` up by *either* growing the
  σReparam γ or by spreading the singular spectrum of the underlying
  raw weight — both are valid responses to the floor and both improve
  the network's representational capacity.
* No double-counting between σReparam and rank-floor: σReparam fixes
  σ_max, rank-floor pushes ``‖W_eff‖_F`` *up* until the ratio satisfies
  the floor. The two interventions sit on orthogonal degrees of freedom.

σ_max is estimated with a 2-step power iteration on the runtime weight
(detached). The gradient flows only through ``‖W‖_F²`` — pushing it
*up* — not through σ_max. This is intentional: we don't want the
optimiser to game the loss by *shrinking* the top singular value, only
by *growing* the rest of the spectrum.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _power_iter_sigma_max(W: torch.Tensor, n_iters: int = 2) -> torch.Tensor:
    """Estimate σ_max(W) via power iteration. Returns a *detached* scalar."""
    with torch.no_grad():
        W = W.detach()
        if W.ndim != 2:
            raise ValueError(f"power-iter expects 2D weight, got ndim={W.ndim}")
        d_out, d_in = W.shape
        v = torch.randn(d_in, device=W.device, dtype=W.dtype)
        v = v / (v.norm() + 1e-8)
        for _ in range(n_iters):
            u = W @ v
            u = u / (u.norm() + 1e-8)
            v = W.T @ u
            v = v / (v.norm() + 1e-8)
        sigma = (W @ v).norm()
    return sigma


class RankFloorRegularizer:
    """Soft penalty for ``stable_rank(W) < target_stable_rank`` per linear.

    Construction
    ------------
    Pass the model and a regex filter for which submodules to regulate.
    Default filter ``"decoder|lidar_prior"`` covers everything the
    K6/T1/T10/T11 post-mortems flagged: decoder cross-attn, decoder
    self-attn, FFN, the V3 shared updater, the V2 prior layer.

    Forward
    -------
    Call ``compute_loss()`` once per training step *after* the main
    forward. It snapshots the runtime ``linear.weight`` of every
    target linear, computes σ_max via power iteration (detached),
    computes ``‖W‖_F²`` on-graph, and returns the sum of squared
    shortfalls scaled by ``self.weight``.

    Notes
    -----
    * **No state.** This is intentionally *not* an ``nn.Module`` so it
      doesn't pollute checkpoints or hparams. Call ``compute_loss``
      from the Lightning module's ``training_step``.
    * **σReparam-aware.** When σReparam is active, ``linear.weight``
      already returns the parametrised view (γ · W_raw / σ̂). The
      regulariser sees that view, so the floor naturally targets the
      *effective* spectrum the network actually uses.
    * **No-op cost when ``weight=0``.** The constructor still
      enumerates targets so ``num_target_linears`` is correct for
      logging, but ``compute_loss`` returns a zero tensor without
      iterating.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_stable_rank: float = 64.0,
        weight: float = 1e-3,
        target_module_filter: str = r"decoder|lidar_prior",
        n_power_iter: int = 2,
        min_dim: int = 64,
    ):
        if target_stable_rank <= 0:
            raise ValueError(
                f"target_stable_rank must be positive, got {target_stable_rank}"
            )
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")

        self.target = float(target_stable_rank)
        self.weight = float(weight)
        self.n_power_iter = int(n_power_iter)
        self._filter = re.compile(target_module_filter)
        self._min_dim = int(min_dim)

        self._linears: List[Tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self._filter.search(name):
                continue
            # Skip tiny linears whose ``min(out, in)`` is below the
            # floor — clamping their stable_rank to 64 would be
            # ill-defined.
            if min(module.weight.shape) < self._min_dim:
                continue
            self._linears.append((name, module))

        logger.info(
            f"RankFloorRegularizer: tracking {len(self._linears)} linears "
            f"(filter={target_module_filter!r}, target_stable_rank="
            f"{self.target:.1f}, weight={self.weight:g}, min_dim={self._min_dim})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def num_target_linears(self) -> int:
        return len(self._linears)

    def target_names(self) -> Iterable[str]:
        return (name for name, _ in self._linears)

    def compute_loss(self) -> torch.Tensor:
        """Return ``self.weight · Σ_l max(0, target − stable_rank_l)²``."""
        if not self._linears or self.weight == 0.0:
            # Build an on-device zero so callers can do `loss = main + reg`
            # without dtype/device juggling.
            ref = self._linears[0][1].weight if self._linears else None
            if ref is None:
                return torch.zeros((), dtype=torch.float32)
            return torch.zeros((), device=ref.device, dtype=torch.float32)

        device = self._linears[0][1].weight.device
        total = torch.zeros((), device=device, dtype=torch.float32)

        for _name, linear in self._linears:
            W = linear.weight  # parametrised runtime view if σReparam active
            sigma_max = _power_iter_sigma_max(W, n_iters=self.n_power_iter)
            frob_sq = (W.float() ** 2).sum()
            stable_rank = frob_sq / (sigma_max.float() ** 2 + 1e-12)
            shortfall = (self.target - stable_rank).clamp(min=0.0)
            total = total + shortfall ** 2

        return self.weight * total

    @torch.no_grad()
    def stable_rank_summary(self) -> dict:
        """Return per-linear stable_rank for diagnostics. Detached, CPU floats."""
        out = {}
        for name, linear in self._linears:
            W = linear.weight.detach()
            sigma_max = _power_iter_sigma_max(W, n_iters=self.n_power_iter)
            frob_sq = (W.float() ** 2).sum()
            stable_rank = float(frob_sq / (sigma_max.float() ** 2 + 1e-12))
            out[name] = stable_rank
        return out
