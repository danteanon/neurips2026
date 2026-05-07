"""CATT — CHM-Aligned Token Training.

A bespoke self-supervised objective for the CHM encoder, designed for the
specific structure of this project (paired clean / corrupted CHM, B=8 batches,
spatial alignment between encoder tokens and pixel-level CHM target). It
replaces VICReg as the CHM-token quality regulariser; see
``docs/chm/insights/cross_run.md`` for the multi-batch encoder probe and
calibration audit that motivated the switch.

Why a custom objective
----------------------
VICReg's variance/covariance terms estimate per-channel statistics over the
batch axis and assume ``B ≫ D`` for those estimates to be both unbiased and
full-rank. Our pipeline is locked to ``B = 8`` while ``D = 1024``: at that
scale the covariance matrix is rank-deficient (rank ≤ 7) and the variance
hinge is dominated by sampling noise. Empirically this produced two distinct
failure modes:

* W0/W1: input-blindness / image-level collapse — different images mapped to
  identical token banks because the expander quickly satisfied its loss in a
  trivial subspace.
* B4/W7: degenerate small-magnitude attractor — over-pressured invariance
  drove every token toward 0 and, by construction, all variance/covariance
  statistics collapsed with it.

CATT sidesteps both by **not relying on batch statistics at all**. Every
term is per-token-pair or per-pixel; the supervised regression target
naturally diversifies tokens and the cosine-only consistency term cannot
incentivise magnitude collapse.

Three terms
-----------

**(1) Per-token local CHM regression.**

For each token ``z[b, t, :]`` produced by the CHM encoder, predict the
average clean CHM value over the patch the token corresponds to::

    pred[b, t]   = LocalCHMHead(z[b, t, :])               # [B, T, k*k]
    target[b, t] = avg_pool2d(chm_clean, kernel=patch//k) at token t  # [B, T, k*k]

    L_local = L1(pred, target)

This is supervised, batch-size-independent, and ties each token's content to
a known geometric anchor — the encoder cannot satisfy it by collapsing to a
constant or by ignoring the input.

**(2) Directional cross-view consistency.**

Two independent corruptions of the same clean source produce ``z1, z2`` of
shape ``[B, T, D]``. The loss aligns the *direction* of paired tokens, not
their magnitude::

    z1_n = F.normalize(z1, dim=-1)
    z2_n = F.normalize(z2, dim=-1)
    L_cons = (1 − (z1_n · z2_n).sum(-1)).mean()           # ∈ [0, 2]

Cosine-only is the load-bearing design choice — VICReg's ``MSE(z1, z2)``
invariance term collapses magnitudes whenever it dominates, because moving
``z1, z2 → 0`` reaches loss-zero. ``1 − cos`` is scale-invariant; the
encoder cannot reduce it by shrinking outputs.

(Token-level reconstruction of the full CHM map — equivalent to a denoising
autoencoder with the *clean* CHM as target — is intentionally **not** in
this module: the existing :class:`model.dinov3_height_model.CHMReconHead`
already implements that head, and CATT runs are expected to enable it via
``aux_chm_recon: true`` + ``aux_chm_recon_target: clean`` +
``aux_chm_recon_weight > 0``. Splitting the recon term out keeps the head
co-located with the rest of the model architecture.)

Config surface
--------------
.. code-block:: yaml

    chm_catt:
      enabled: true
      patch_size: 16              # CHM-encoder downsampling factor (1 token = 16×16 px)
      sub_patch_k: 1              # 1 → predict mean per token; 4 → predict 4×4 grid
      lambda_local: 1.0           # weight on per-token regression
      lambda_consistency: 1.0     # weight on directional consistency
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["LocalCHMHead", "CATTLoss"]


class LocalCHMHead(nn.Module):
    """Linear head predicting per-token local CHM.

    Maps each ``D``-dim token to a ``k × k`` grid of CHM values covering the
    token's spatial receptive field. ``k = 1`` is the simplest version (one
    mean CHM value per token); ``k > 1`` exposes finer within-patch
    structure. Total output dimension is ``k * k`` per token.

    The head is intentionally a single ``Linear`` — this is the *probe*
    in "linear probing": if a linear map cannot read CHM out of the
    encoder's tokens, the encoder is not encoding useful CHM information.
    Adding hidden layers would let the head paper over a weak encoder.
    """

    def __init__(self, embed_dim: int, sub_patch_k: int = 1):
        super().__init__()
        if sub_patch_k < 1:
            raise ValueError(
                f"sub_patch_k must be >= 1, got {sub_patch_k}"
            )
        self.embed_dim = embed_dim
        self.sub_patch_k = sub_patch_k
        self.linear = nn.Linear(embed_dim, sub_patch_k * sub_patch_k)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: ``[B, T, D]`` CHM-encoder token bank.

        Returns:
            ``[B, T, k*k]`` per-token CHM predictions.
        """
        return self.linear(tokens)


class CATTLoss(nn.Module):
    """CHM-Aligned Token Training loss.

    Owns the :class:`LocalCHMHead` and computes the two CATT terms whose
    targets are ``chm_clean`` and the two corrupted views::

        L_CATT = λ_local · L_local(z, chm_clean)
               + λ_cons  · L_cons(z1, z2)

    The encoder is invoked **outside** this module by the caller (the
    LightningModule) — we operate purely on tokens here so we never have to
    know which encoder variant is in use.

    Token / pixel alignment
    -----------------------
    The CHM encoder downsamples by ``patch_size`` (default 16). For an
    ``H × W`` CHM, the token grid is ``(H/patch_size) × (W/patch_size)``. We
    pool ``chm_clean`` to a ``(h_grid · k) × (w_grid · k)`` map, then split
    each ``k × k`` block into the per-token target. With ``k = 1`` this is
    simply ``avg_pool2d(chm_clean, patch_size)``.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 16,
        sub_patch_k: int = 1,
        lambda_local: float = 1.0,
        lambda_consistency: float = 1.0,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if sub_patch_k < 1:
            raise ValueError(f"sub_patch_k must be >= 1, got {sub_patch_k}")
        if patch_size % sub_patch_k != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by "
                f"sub_patch_k ({sub_patch_k})"
            )
        self.patch_size = patch_size
        self.sub_patch_k = sub_patch_k
        self.lambda_local = float(lambda_local)
        self.lambda_consistency = float(lambda_consistency)
        self.local_head = LocalCHMHead(embed_dim=embed_dim, sub_patch_k=sub_patch_k)

    @staticmethod
    def _local_target(
        chm_clean: torch.Tensor, patch_size: int, sub_patch_k: int
    ) -> torch.Tensor:
        """Pool clean CHM to per-token ``k × k`` targets.

        Args:
            chm_clean: ``[B, 1, H, W]`` clean CHM.
            patch_size: encoder downsampling factor.
            sub_patch_k: sub-patches per token side.

        Returns:
            ``[B, T, k*k]`` target with ``T = (H/patch) · (W/patch)`` and
            ``k = sub_patch_k``.
        """
        B, C, H, W = chm_clean.shape
        if C != 1:
            raise ValueError(
                f"Expected single-channel CHM, got {C} channels"
            )
        if H % patch_size or W % patch_size:
            raise ValueError(
                f"CHM size ({H}×{W}) must be divisible by patch_size "
                f"({patch_size})"
            )
        # Pool to (h_grid · k) × (w_grid · k) so each token covers a k × k
        # block of pooled values.
        sub_kernel = patch_size // sub_patch_k
        pooled = F.avg_pool2d(chm_clean, kernel_size=sub_kernel)  # [B, 1, h·k, w·k]
        h_grid = H // patch_size
        w_grid = W // patch_size
        # Group per-token: [B, 1, h_grid, k, w_grid, k] → [B, h_grid, w_grid, k, k]
        pooled = pooled.view(
            B, 1, h_grid, sub_patch_k, w_grid, sub_patch_k
        ).permute(0, 2, 4, 3, 5, 1).contiguous()
        # [B, h_grid, w_grid, k, k] → [B, T, k*k]
        return pooled.view(B, h_grid * w_grid, sub_patch_k * sub_patch_k)

    def local_loss(
        self,
        tokens: torch.Tensor,
        chm_clean: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """L1 between :class:`LocalCHMHead` predictions and pooled clean CHM.

        Args:
            tokens: ``[B, T, D]`` encoder tokens.
            chm_clean: ``[B, 1, H, W]`` clean CHM.
            valid_mask: optional ``[B]`` bool. ``True`` means the sample
                contributes to the loss. Excluded samples cover the case
                where dataset-level full_dropout zeroed the input — there
                is no meaningful CHM to predict for those.

        Returns:
            scalar L1 loss.
        """
        target = self._local_target(
            chm_clean, self.patch_size, self.sub_patch_k
        )                                                       # [B, T, k*k]
        pred = self.local_head(tokens)                          # [B, T, k*k]

        if valid_mask is not None:
            if valid_mask.sum() == 0:
                return torch.zeros((), device=tokens.device, dtype=tokens.dtype)
            pred = pred[valid_mask]
            target = target[valid_mask]
        return F.l1_loss(pred, target)

    @staticmethod
    def consistency_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Directional consistency between two views, per token.

        Cosine-only (scale-invariant) so the encoder cannot reduce it by
        shrinking magnitudes — that is the failure mode VICReg's MSE
        invariance produced in B4. The loss is bounded in ``[0, 2]`` with
        ``0`` at perfect alignment and ``1`` at orthogonality.

        Args:
            z1, z2: ``[B, T, D]`` token banks from two corrupted views.
            eps: numerical floor on token norms.
        """
        z1_n = F.normalize(z1, dim=-1, eps=eps)
        z2_n = F.normalize(z2, dim=-1, eps=eps)
        cos = (z1_n * z2_n).sum(dim=-1)                          # [B, T]
        return (1.0 - cos).mean()

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        chm_clean: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the full CATT loss on a pair of views.

        Per-token regression is run on the average of the two views' tokens
        (``(z1 + z2) / 2``) so it sees both corruption realisations and the
        gradient flows through both encoder branches with equal weight. We
        explicitly do not pick one view at random — that would halve the
        effective gradient signal each step. Using the mean is also
        consistent with the consistency term's symmetric treatment of the
        two views.

        Args:
            z1, z2: ``[B, T, D]`` encoder tokens from view 1, view 2.
            chm_clean: ``[B, 1, H, W]`` clean CHM (target for the
                regression term).
            valid_mask: optional ``[B]`` bool — same semantics as in
                :meth:`local_loss`.

        Returns:
            ``(loss, components)``. ``components`` carries the per-term
            scalars (``local``, ``consistency``) for diagnostic logging,
            all detached.
        """
        if z1.shape != z2.shape:
            raise ValueError(
                f"z1 and z2 must have identical shape, got {tuple(z1.shape)} "
                f"vs {tuple(z2.shape)}"
            )

        z_avg = 0.5 * (z1 + z2)
        l_local = self.local_loss(z_avg, chm_clean, valid_mask=valid_mask)
        l_cons = self.consistency_loss(z1, z2)

        loss = self.lambda_local * l_local + self.lambda_consistency * l_cons
        components = {
            "local": l_local.detach(),
            "consistency": l_cons.detach(),
        }
        return loss, components
