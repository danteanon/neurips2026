"""VICReg loss for the CHM encoder, calibrated against the official code.

Reference paper: Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-
Covariance Regularization for Self-Supervised Learning." ICLR 2022
(arXiv:2105.04906).

Reference code: https://github.com/facebookresearch/vicreg
(``main_vicreg.py`` lines 184–220 are the canonical loss).

This module provides two pieces:

1. :class:`VICRegExpander` — the 3-layer MLP "expander" `h_φ` from the
   paper, which projects the encoder representation `Y` into a wider
   embedding space `Z` where the variance / covariance hinges actually
   carry useful gradient. The expander is **mandatory** for the paper's
   loss formulation to work; running VICReg directly on encoder
   features at native dimension and without BN-induced whitening leaves
   the variance and covariance terms with nothing they can meaningfully
   push around.

2. :class:`VICRegLoss` — the three-term loss, taking per-image
   embeddings ``[N, D]`` (one row per image) and computing the **same
   reduction the official code uses**, which differs by constant
   factors from the paper equations:

     s_code(Z, Z') = (1 / (N · D)) Σ_n Σ_d (z_nd − z'_nd)²
                   = F.mse_loss(Z, Z')
                   ≡ (1 / D) · s_paper(Z, Z')           # paper Eq. (5) / D

     v(Z)         = (1 / D) Σ_d max(0, γ − √(Var(z_d) + ε))
                                                          # paper Eq. (1)

     c(Z)         = (1 / D) Σ_{i ≠ j} [C(Z)]_{ij}²        # paper Eq. (3, 4)

   with ``C(Z) = (1 / (N − 1)) Z_centered^T Z_centered`` and per-channel
   variance taken over the batch axis. Total loss (mirroring
   ``main_vicreg.py:215–219``, including the ``/2`` on variance that
   the paper equations leave out):

     L = λ · s_code(Z, Z')
       + μ · [v(Z) + v(Z')] / 2
       + ν · [c(Z) + c(Z')]

   Why match the code, not the equations: the paper's stated defaults
   ``λ = μ = 25, ν = 1`` are calibrated against the *code* (that is
   what produced the released checkpoints). Matching the equations
   literally and reusing those defaults would put ``D ≈ 2048×`` more
   invariance pressure and ``2×`` more variance pressure than the
   reference implementation. See ``docs/chm/insights/cross_run.md``
   for the audit and the B4/W7 regression that motivated this change.

Adaptation for our setup
------------------------
The paper's encoder produces one D-dim feature per image (ResNet-50 →
2048-dim ``Y``). Our CHM encoder produces patch-level features
``[B, T, D] = [B, 256, 1024]``. To stay paper-faithful, we **mean-pool
tokens over the spatial axis** before the expander:

    chm_tokens [B, T, D]
        │  mean over T
        ▼
    [B, D]
        │  expander h_φ
        ▼
    [B, D_exp]
        │  VICRegLoss
        ▼
    L

Mean-pooling discards within-image diversity for the purpose of this
loss; the loss anchors against image-to-image collapse (different
images must produce different per-channel values), which is exactly
what was missing from the previous adaptation and led to the
input-blindness collapse observed in W0/W1.

Hyperparameter defaults
-----------------------
The official defaults (γ = 1, λ = μ = 25, ν = 1, ε = 1e-4) are used
verbatim and are now applied to the **same reductions the official
code uses** (see module-level summary above). They are calibrated for
BN-stabilised expander outputs; our expander is also BN-stabilised so
the calibration carries over. Batch size is application-specific
(official 1000-epoch run uses 2048, we use 32–64); per-channel std
estimation noise is the only sample-size-dependent quantity and
remains tolerable down to B = 32 (see paper Table 13, B ≥ 128 is the
ablated regime).

Deviations from the official setup
----------------------------------
* **Encoder shape.** Official: image-level ResNet-50 (one ``[D]`` vec
  per image, ``D = 2048``). Us: patch-level CNN, mean-pooled across
  tokens before the expander (or per-token, if granularity = "token").
* **Expander width.** Official: ``mlp = "8192-8192-8192"`` (output
  ``D_exp = 8192``). Us: ``expander_dim = 2048`` by default; paper
  Table 12 shows 2048 is enough at our scale.
* **Augmentations.** Official: SimCLR-style RGB. Us: CHM-specific
  alignment-preserving (noise, blur, scale jitter).
* **Optimizer / batch / epochs.** Official: LARS, batch 2048, 1000
  epochs. Us: Muon/AdamW, batch 32, ~50 epochs joint with the
  supervised height task. Smaller batch is the most consequential —
  paper Table 13 ablates down to B = 128 with modest degradation; B =
  32 is below the paper-tested regime and per-channel std estimation
  is correspondingly noisier (~8× standard error vs B = 2048).
* **Expander persistence.** Official: discarded after pretraining.
  Us: retained inside the Lightning module but only consumed by this
  loss; the downstream height-prediction path does not see the
  expander output.
* **DDP gather.** Official: ``FullGatherLayer`` concatenates all DDP
  ranks before computing variance/covariance. Us: local-batch only
  (single-GPU equivalent; in multi-GPU training this would compute
  per-rank statistics and is a known calibration drift to revisit if
  we ever scale out).
* **Final projector layer bias.** Official: ``Linear(..., bias=False)``
  on the last expander layer. Us: default ``bias=True``. Mathematically
  inconsequential (a constant shift that cancels in MSE / variance /
  covariance), kept for simplicity.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class VICRegExpander(nn.Module):
    """Paper's 3-layer expander ``h_φ``.

    Architecture (paper Section 4.2): two FC layers with BN + ReLU,
    plus a third plain Linear layer. All hidden / output dimensions
    are ``expander_dim``.

        ``Linear(in_dim → expander_dim) → BN → ReLU
         → Linear(expander_dim → expander_dim) → BN → ReLU
         → Linear(expander_dim → expander_dim)``

    Args:
        in_dim:        Input feature dimension (encoder representation
                       width, e.g. 1024 for our CHM encoder).
        expander_dim:  Hidden / output width. Paper uses 8192; the
                       Top-1 vs ``D`` study (paper Appendix D.6,
                       Table 12) shows 2048 is enough at our scale —
                       4 × less compute for ~1 point of paper-protocol
                       Top-1 we are not optimising for.
        num_layers:    Number of Linear layers. Paper uses 3 (default
                       here). 2 also works empirically and saves
                       ~``expander_dim²`` params.
    """

    def __init__(
        self,
        in_dim: int,
        expander_dim: int = 2048,
        num_layers: int = 3,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(
                f"VICRegExpander needs at least 2 Linear layers, got {num_layers}"
            )

        self.in_dim = in_dim
        self.expander_dim = expander_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            d_in = in_dim if i == 0 else expander_dim
            layers.append(nn.Linear(d_in, expander_dim))
            layers.append(nn.BatchNorm1d(expander_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(expander_dim, expander_dim))
        self.expander = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Project per-image features through the expander.

        Args:
            y: ``[N, in_dim]`` per-image representations.

        Returns:
            ``[N, expander_dim]`` expanded embeddings ready for the
            VICReg loss.
        """
        if y.ndim != 2:
            raise ValueError(
                f"VICRegExpander expects [N, D] input, got shape {tuple(y.shape)}"
            )
        return self.expander(y)


class VICRegLoss(nn.Module):
    """VICReg loss on per-image embeddings ``[N, D]``.

    The reduction matches the official ``main_vicreg.py`` reference
    implementation (``F.mse_loss`` for invariance, ``/2`` averaging on
    the two-branch variance hinge, sum on the two-branch covariance).
    Inputs are expected to be **expander outputs** (after
    :class:`VICRegExpander`); applying VICReg directly to raw encoder
    features under-performs (see module docstring).

    Args:
        gamma:       Per-channel std target in the variance hinge
                     ``relu(γ − std_d)``. Paper default is 1.0,
                     calibrated for BN-stabilised expander outputs.
        lambda_inv:  Weight on the invariance term, applied to
                     ``F.mse_loss(z1, z2)`` (mean over both N and D).
                     Paper / official default 25.
        lambda_var:  Weight on the two-branch variance hinge, applied
                     to ``(v(Z) + v(Z')) / 2``. Paper / official
                     default 25.
        lambda_cov:  Weight on the off-diagonal covariance term,
                     applied to ``c(Z) + c(Z')``. Paper / official
                     default 1.
        eps:         Numerical stabiliser inside ``sqrt(var + eps)``.
                     Paper / official default 1e-4.

    Forward signature: ``loss, components = vicreg(z1, z2)``
        ``z1, z2`` are ``[N, D]`` expander outputs from the two views,
        ``components`` is a dict of unweighted detached scalars
        ``{"inv", "var", "cov"}`` for logging. ``var`` is the
        averaged-over-branches hinge (matches the value
        ``main_vicreg.py:207`` would log); ``cov`` is the summed
        two-branch term (matches ``main_vicreg.py:211–213``).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        eps: float = 1.0e-4,
    ):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        for name, val in (
            ("lambda_inv", lambda_inv),
            ("lambda_var", lambda_var),
            ("lambda_cov", lambda_cov),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        self.gamma = float(gamma)
        self.lambda_inv = float(lambda_inv)
        self.lambda_var = float(lambda_var)
        self.lambda_cov = float(lambda_cov)
        self.eps = float(eps)

    @staticmethod
    def _invariance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Invariance term, matching the official VICReg implementation.

        Reference:
        ``main_vicreg.py:198`` →  ``repr_loss = F.mse_loss(x, y)``,
        which averages over **both** N and D:
        ``F.mse_loss(z1, z2) = (1 / (N · D)) Σ_n Σ_d (z_nd − z'_nd)²``.

        The paper's Eq. (5) writes the invariance as the per-row L2²
        distance averaged over the batch
        ``s(Z, Z') = (1/N) Σ_n ||z_n − z'_n||²``,
        which differs from the code form by a factor of ``D``. The
        published default ``λ_inv = 25`` is calibrated against the
        **code form** — that is what produced the released checkpoints
        — so this implementation deliberately follows the code rather
        than the paper equation.

        History: an earlier "paper-faithful" rewrite used
        ``((z1 - z2) ** 2).sum(dim=1).mean()`` which is the paper Eq.
        (5) form. With ``λ_inv = 25`` and ``D = 2048`` that put
        roughly ``D × ≈ 2048×`` more invariance pressure on the encoder
        than the official code applies, which contributed to the rank
        / variance regression observed in B4/W7. See
        ``docs/chm/insights/cross_run.md`` for the audit.
        """
        return ((z1 - z2) ** 2).mean()

    def _variance(self, z: torch.Tensor) -> torch.Tensor:
        """Per-branch variance hinge, paper Eq. (1) form.

        ``v(Z) = (1/D) Σ_d max(0, γ − √(Var(z_d) + ε))``

        Std is taken over ``dim=0`` (the batch) — one std value per
        channel, ``D`` values total, then ``relu(γ − std_d)`` applied
        per channel and averaged over D.

        The official code averages the two branches in
        :meth:`forward` (``std_loss = mean(...)/2 + mean(...)/2``),
        so the per-branch quantity here is exactly Eq. (1).
        """
        std = torch.sqrt(z.var(dim=0, unbiased=True) + self.eps)        # [D]
        return torch.relu(self.gamma - std).mean()

    def _covariance(self, z: torch.Tensor) -> torch.Tensor:
        """Paper Eq. (3, 4): off-diagonal squared sum of cov / D.

        ``c(Z) = (1 / D) Σ_{i ≠ j} [C(Z)]_{ij}²``
        with
        ``C(Z) = (1 / (N − 1)) Z_centered^T Z_centered``.

        Note: this is the **raw** covariance, not a correlation —
        we deliberately do not standardise per-channel before the
        Gram product. The paper specifically tested standardisation
        (Table 8) and found it slightly hurts (-0.2% Top-1).
        """
        N, D = z.shape
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(N - 1, 1)                               # [D, D]
        sq = cov.pow(2)
        diag_sq = sq.diagonal().sum()
        total_sq = sq.sum()
        off_sq = total_sq - diag_sq
        return off_sq / D

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the VICReg loss on two per-image embedding banks.

        Args:
            z1, z2: ``[N, D]`` expander outputs from the two augmented
                views. Must have matching shape.

        Returns:
            ``(loss, components)``. ``loss`` is the weighted total;
            ``components`` is a dict of unweighted detached scalars
            for logging.
        """
        if z1.shape != z2.shape:
            raise ValueError(
                f"VICReg expects matching shapes, got {tuple(z1.shape)} vs {tuple(z2.shape)}"
            )
        if z1.ndim != 2:
            raise ValueError(
                f"VICReg expects [N, D] inputs, got ndim={z1.ndim}"
            )

        inv = self._invariance(z1, z2)
        # Match official ``main_vicreg.py:207``:
        #   ``std_loss = mean(relu(1 - std_x)) / 2 + mean(relu(1 - std_y)) / 2``
        # i.e. the AVERAGE of the two per-branch hinges, not the sum.
        # The published default ``λ_var = 25`` is calibrated against this
        # form. Using a sum would put 2× the variance pressure the
        # released checkpoints were trained with.
        var = (self._variance(z1) + self._variance(z2)) / 2.0
        # Covariance composition does match the official code (sum of
        # two per-branch terms); see ``main_vicreg.py:211–213``.
        cov = self._covariance(z1) + self._covariance(z2)

        total = (
            self.lambda_inv * inv
            + self.lambda_var * var
            + self.lambda_cov * cov
        )
        return total, {
            "inv": inv.detach(),
            "var": var.detach(),
            "cov": cov.detach(),
        }
