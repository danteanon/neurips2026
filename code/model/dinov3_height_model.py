"""CHM-Prompted Height Estimation with a proper Transformer decoder stack.

High-level architecture (top-down):

    Image   [B, 3, H, W]  ──► Dinov3Backbone (frozen)  ──► img_tokens  [B, P, D]

    CHM     [B, 1, H, W]  ──► CHMPromptEncoder (CNN)   ──► chm_tokens  [B, P', D]
                                                       ──► + 2D sin PE ──► chm_memory

    img_tokens, chm_memory ──► HeightDecoderStack (N layers, default 10)
                            │   Each layer:
                            │     Sub-block 1:  x = x + MHA( LN(x), chm_memory, mask )  [cross-attn, windowed]
                            │     Sub-block 2:  x = x + MHA( LN(x), LN(x)           )   [self-attn,  global  ]
                            │     Sub-block 3:  x = x + FFN( LN(x) )                    [ffn                 ]
                            └► refined_tokens  [B, P, D]

    Cross-attention uses a **locality prior**: query ``(y_q, x_q)`` may only attend
    to memory tokens inside a ``(2k+1) × (2k+1)`` window centered at ``(y_q, x_q)``.
    The mask is ``[P, P]`` bool, cached per grid size, and applied additively as
    ``-inf`` before softmax. Self-attention stays global.

    refined_tokens ──► reshape [B, D, h, w] ──► SimpleConvHead ──► height_map  [B, 1, H, W]

Each sub-block is its own ``nn.Module`` so you can inspect inputs / outputs at any
depth.  ``forward(..., return_intermediates=True)`` returns a dict of intermediate
tensors for every layer and sub-block.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_model import Dinov3Backbone
from .depth_dpt import DPTHeadLinear


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------


class _PostScale(nn.Module):
    """Multiply the (already-σ-capped) weight by a fixed scalar ``scale``.

    Stacked on top of ``spectral_norm`` so the composed forward is
    ``scale * (W / σ)``. Effective σ_max of the parametrised weight is
    ``scale``. Implements ``right_inverse`` so that ``load_state_dict``
    and ``module.weight = ...`` round-trip correctly through
    ``parametrize.register_parametrization``.
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.scale

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        return w / self.scale


class _SigmaReparamScale(nn.Module):
    """Learnable post-scale γ, the σReparam recipe (Zhai et al. 2023, eq. 2).

    Stacked on top of ``spectral_norm`` so the composed forward is
    ``γ · (W / σ(W))``, hence the *effective* σ_max of the parametrised
    weight equals exactly ``γ``. ``γ`` is an ``nn.Parameter`` (one scalar
    per linear layer), initialised to ``init`` (default 1.0 — the paper's
    init), and learned by the optimizer.

    Why learnable matters
    ---------------------
    Vanilla ``spectral_norm`` (Miyato et al. 2018) hard-pins σ_max at 1
    and removes that degree of freedom from the optimizer. The σReparam
    paper proves (Proposition 3.2) that the ideal Adam update has
    spectral norm ~√w for a width-w matrix, so a fixed σ=1 cap discards
    most of the optimizer's signal. Replacing the hard cap with a
    learnable scalar γ keeps the dimensionality-independent update
    dynamics while still preventing entropy collapse (Theorem 3.1).
    Empirically, the paper's Table 1 / Table 7 measure a ~12-point
    accuracy gap between SN (γ=1 fixed, 69.81% ImageNet) and σReparam
    (γ learnable, 82.2%).
    """

    def __init__(self, init: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(init)))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.gamma * w

    def right_inverse(self, w: torch.Tensor) -> torch.Tensor:
        # Use detach to avoid accidentally tracking γ in a setter context
        # (load_state_dict, weight assignment). γ itself remains learnable
        # via the forward path.
        return w / self.gamma.detach().clamp(min=1e-6)


_VALID_GAMMA_INIT_MODES = ("constant", "svd")


def _apply_spectral_reparam(
    proj: nn.Linear,
    *,
    learnable: bool = False,
    init: float = 1.0,
    gamma_init_mode: str = "constant",
    n_power_iterations: int = 1,
) -> None:
    """Apply spectral normalisation to ``proj.weight`` with three variants.

    ============================  ===========  ================================
    ``(learnable, init)``          σ_max of    Equivalent to
                                   ``W_eff``
    ============================  ===========  ================================
    ``(False, 1.0)``              ``= 1``      Vanilla SN (Miyato 2018)
    ``(False, c≠1)``              ``= c``      SN + fixed post-scale (T9/T10)
    ``(True, init=c)``            ``= γ``      σReparam paper (Zhai 2023)
                                   (γ init=c,
                                   learnable)
    ============================  ===========  ================================

    ``gamma_init_mode`` (only meaningful when ``learnable=True``):

    * ``"constant"`` — γ is initialised to ``init`` (default 1.0). Matches
      the σReparam paper's Table 13 NLP recipe and our T11/T12/T13
      runs. Best when ``σ(W_init) ≈ 1`` (e.g. trunc-normal-0.02 init).
    * ``"svd"`` — γ is initialised to ``σ_max(W_init)``: the *largest
      singular value of the linear's freshly initialised weight*.
      Matches the σReparam paper's vision recipe (§4.2: "In vision we
      initialize the σReparam γ term using the first singular value")
      and the recommendation distilled from the T11 post-mortem
      (`U_series_stabilization_plan.md` §3.0).

      Why this matters: PyTorch's default ``nn.Linear`` init has
      ``σ(W_init) ≈ √(in/3)`` ≫ 1 for ``D=1024``, so γ=1 starts the
      forward at *much* smaller activation magnitude than a plain Linear
      would, giving the optimiser nothing to learn from at iter 0
      (T11's empirical observation: γ moved < 2% over 3 epochs).
      γ_init = σ_max(W) recovers the at-init forward magnitude exactly,
      and any learning then rebalances spectrum *shape* without first
      having to grow γ scalar-wise.

    For the joint-QKV layout: ``proj`` is the *fused* projection
    (``Linear(D, 3D)`` for self-attn, ``Linear(D, D)`` and
    ``Linear(D, 2D)`` for cross-attn Q and KV). Capping the fused
    weight's σ_max is strictly tighter than per-Q/K/V caps and removes
    the gradient "reservoir" route through any single uncapped slice
    (the V-side rank-1 collapse mechanism we hit in T8).
    """
    from torch.nn.utils.parametrizations import spectral_norm
    from torch.nn.utils import parametrize

    if gamma_init_mode not in _VALID_GAMMA_INIT_MODES:
        raise ValueError(
            f"gamma_init_mode must be one of {_VALID_GAMMA_INIT_MODES}, "
            f"got {gamma_init_mode!r}"
        )
    if gamma_init_mode == "svd" and not learnable:
        raise ValueError(
            "gamma_init_mode='svd' requires learnable=True. "
            "A non-learnable γ uses the static `_PostScale` parametrisation "
            "where γ is fixed; SVD-from-W_init only makes sense when γ is "
            "an `nn.Parameter`. Set sigma_reparam_learnable=true."
        )

    # Compute the SVD-based init *before* registering ``spectral_norm``: the
    # parametrisation replaces ``proj.weight`` with the σ-normalised view, so
    # ``proj.weight`` after registration no longer reads the raw init we
    # need. Run SVD on a CPU float32 copy for deterministic numerics.
    if learnable and gamma_init_mode == "svd":
        with torch.no_grad():
            sigma_max = torch.linalg.svdvals(
                proj.weight.detach().to(torch.float32)
            ).max()
        gamma_init_value = float(sigma_max.item())
    else:
        gamma_init_value = float(init)

    spectral_norm(proj, name="weight", n_power_iterations=int(n_power_iterations))

    if learnable:
        parametrize.register_parametrization(
            proj, "weight", _SigmaReparamScale(gamma_init_value)
        )
    elif gamma_init_value != 1.0:
        parametrize.register_parametrization(
            proj, "weight", _PostScale(gamma_init_value)
        )


def _apply_joint_qkv_spectral_norm(
    proj: nn.Linear,
    sigma_cap: float = 1.0,
    n_power_iterations: int = 3,
    *,
    learnable: bool = False,
    gamma_init_mode: str = "constant",
) -> None:
    """Backwards-compatible wrapper around :func:`_apply_spectral_reparam`.

    Kept for legacy call sites that pass ``sigma_cap`` positionally. New
    code should call :func:`_apply_spectral_reparam` directly.
    """
    _apply_spectral_reparam(
        proj,
        learnable=learnable,
        init=float(sigma_cap),
        gamma_init_mode=gamma_init_mode,
        n_power_iterations=int(n_power_iterations),
    )


def apply_sigma_reparam_to_all_linears(
    module: nn.Module,
    *,
    learnable: bool = True,
    init: float = 1.0,
    gamma_init_mode: str = "constant",
    n_power_iterations: int = 1,
    skip_already_parametrised: bool = True,
) -> int:
    """Walk ``module`` recursively and apply σReparam to every ``nn.Linear``.

    Matches the paper's "all linear layers" coverage recipe (Zhai et al.
    2023, §4.1). Returns the count of linears that received the
    parametrisation. By default, layers that *already* carry a
    parametrisation on ``weight`` (e.g. fused QKV projections that the
    per-site path installed first with the joint cap) are skipped, so
    this function never double-wraps; per-site QKV caps win.

    Typical use::

        model.decoder = HeightDecoderStackV3(...)         # QKV caps may already
                                                          # have been installed
        n = apply_sigma_reparam_to_all_linears(            # extends coverage to
            model.decoder, learnable=True, init=1.0)       # out_proj, FFN.fc1,
                                                          # FFN.fc2, etc.
    """
    from torch.nn.utils import parametrize

    count = 0
    for _name, sub in module.named_modules():
        if not isinstance(sub, nn.Linear):
            continue
        if skip_already_parametrised and parametrize.is_parametrized(sub, "weight"):
            continue
        _apply_spectral_reparam(
            sub,
            learnable=learnable,
            init=init,
            gamma_init_mode=gamma_init_mode,
            n_power_iterations=n_power_iterations,
        )
        count += 1
    return count


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention with two stability levers.

    Layout follows the modern fused-projection convention (ViT-22B,
    Llama-3, σReparam):

    * **Self-attention** (``attn_type="self"``): one fused ``qkv_proj`` of
      shape ``Linear(D, 3D)``. Q/K/V are split out of the output along the
      feature dim per token. Saves one matmul vs three separate projections.
    * **Cross-attention** (``attn_type="cross"``): ``q_proj`` is
      ``Linear(D, D)`` (acts on ``q_source``) and ``kv_proj`` is
      ``Linear(D, 2D)`` (acts on ``kv_source``). K and V are split out of
      ``kv_proj``'s output.

    Two opt-in stability levers are wired in (defaults are no-ops, so the
    module is a vanilla pre-LN MHA out of the box):

    * ``qk_norm``: per-head ``LayerNorm`` applied to Q and K *after* the
      head reshape and *before* the dot product (Dehghani et al. 2023,
      ViT-22B). Caps the attention logit magnitude at the LayerNorm gain
      (initialised to 1) so softmax cannot saturate via W_q/W_k spectral
      growth — i.e. it is a *direct* cure for attention-entropy collapse.
      V is intentionally not LN'd: V's per-token magnitude carries
      semantic load in the height decoder, and unit-norming it would
      throw information away.
    * ``qkv_spectral_norm``: register PyTorch's
      :func:`torch.nn.utils.parametrizations.spectral_norm` on the fused
      projection(s). Self-attention gets a *single, joint* cap on
      ``σ_max([W_q; W_k; W_v]) ≤ sigma_cap`` — the strongest σReparam
      variant. Cross-attention caps ``W_q`` and the fused ``[W_k; W_v]``
      separately, which together cap the (block-diagonal)
      cross-attention operator at σ_max ≤ sigma_cap. The argument
      accepts either a ``bool`` (``True`` → σ_cap = 1.0, the original
      Zhai et al. recipe; ``False`` / ``None`` → off) or a positive
      ``float`` that is used directly as σ_cap. Higher caps give the
      network more output gain per projection at the cost of a looser σ
      bound; the K0 historic *V-only* spectral_norm path settled at
      σ ≈ 1.5–2.7 spontaneously, suggesting the joint cap should sit in
      that range or slightly above for cross-attn.

    QK-Norm and joint-QKV-σReparam can be combined; they constrain Q/K
    in different ways (output-side LN vs weight-side spectrum) and both
    leave V's representational range alone (since V is included in the
    σReparam cap but not LN'd).
    """

    VALID_ATTN_TYPES = ("self", "cross")

    @staticmethod
    def _resolve_qkv_spectral_cap(
        qkv_spectral_norm: Union[bool, float, None],
    ) -> Optional[float]:
        """Map the user-facing ``qkv_spectral_norm`` to a σ_cap or ``None``.

        Accepted inputs:

        * ``None`` / ``False`` / ``0`` → no spectral parametrization (returns ``None``).
        * ``True`` → σ_cap = 1.0 (the original Zhai et al. 2023 recipe).
        * Positive ``float`` → use that value as σ_cap directly.

        A non-positive float raises ``ValueError``; an unhandled type also
        raises so the misconfiguration is loud at construction time, not
        silently disabled.
        """
        if qkv_spectral_norm is None or qkv_spectral_norm is False:
            return None
        if qkv_spectral_norm is True:
            return 1.0
        if isinstance(qkv_spectral_norm, (int, float)):
            value = float(qkv_spectral_norm)
            if value == 0.0:
                return None
            if value <= 0.0:
                raise ValueError(
                    "qkv_spectral_norm must be a bool, None, or positive float; "
                    f"got {qkv_spectral_norm!r}"
                )
            return value
        raise TypeError(
            "qkv_spectral_norm must be bool | float | None; "
            f"got {type(qkv_spectral_norm).__name__}"
        )

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        attn_type: str = "cross",
        qk_norm: bool = False,
        qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
    ):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
            )
        if attn_type not in self.VALID_ATTN_TYPES:
            raise ValueError(
                f"attn_type must be one of {self.VALID_ATTN_TYPES}, got {attn_type!r}"
            )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads
        self.scale = math.sqrt(self.d_head)
        self.attn_type = attn_type
        self.qk_norm = bool(qk_norm)

        sigma_cap = self._resolve_qkv_spectral_cap(qkv_spectral_norm)
        self.qkv_spectral_norm = sigma_cap is not None
        self.qkv_spectral_sigma = float(sigma_cap) if sigma_cap is not None else 0.0
        self.qkv_spectral_learnable = bool(qkv_spectral_learnable) and self.qkv_spectral_norm
        self.qkv_spectral_gamma_init = str(qkv_spectral_gamma_init)

        if attn_type == "self":
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
            if self.qkv_spectral_norm:
                _apply_spectral_reparam(
                    self.qkv_proj,
                    learnable=self.qkv_spectral_learnable,
                    init=float(sigma_cap),
                    gamma_init_mode=self.qkv_spectral_gamma_init,
                )
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
            if self.qkv_spectral_norm:
                _apply_spectral_reparam(
                    self.q_proj,
                    learnable=self.qkv_spectral_learnable,
                    init=float(sigma_cap),
                    gamma_init_mode=self.qkv_spectral_gamma_init,
                )
                _apply_spectral_reparam(
                    self.kv_proj,
                    learnable=self.qkv_spectral_learnable,
                    init=float(sigma_cap),
                    gamma_init_mode=self.qkv_spectral_gamma_init,
                )

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self.qk_norm:
            self.q_ln = nn.LayerNorm(self.d_head)
            self.k_ln = nn.LayerNorm(self.d_head)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Diagnostic-only attention cache. When ``capture_attn`` is True the
        # forward stashes the post-softmax attention tensor on
        # ``_last_attn`` (detached, no grad) so external callbacks can
        # compute entropy / max-attention statistics without rerunning the
        # forward. Default False keeps the steady-state memory overhead at
        # zero. Toggled by ``CHMHealthProbes`` before every Nth optimiser
        # step and cleared after consumption.
        self.capture_attn: bool = False
        self._last_attn: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public projection helpers — used by diagnostics that need Q / K / V
    # without the rest of the attention forward (logit recomputation,
    # V-norm probes). They abstract over the self vs cross fusion layout.
    # ------------------------------------------------------------------
    def project_q(self, q_source: torch.Tensor) -> torch.Tensor:
        """Project ``q_source [..., D]`` to Q ``[..., D]`` (pre-head split)."""
        if self.attn_type == "self":
            return self.qkv_proj(q_source)[..., : self.embed_dim]
        return self.q_proj(q_source)

    def project_k(self, kv_source: torch.Tensor) -> torch.Tensor:
        """Project ``kv_source [..., D]`` to K ``[..., D]`` (pre-head split)."""
        D = self.embed_dim
        if self.attn_type == "self":
            return self.qkv_proj(kv_source)[..., D : 2 * D]
        return self.kv_proj(kv_source)[..., :D]

    def project_v(self, kv_source: torch.Tensor) -> torch.Tensor:
        """Project ``kv_source [..., D]`` to V ``[..., D]`` (pre-head split)."""
        D = self.embed_dim
        if self.attn_type == "self":
            return self.qkv_proj(kv_source)[..., 2 * D :]
        return self.kv_proj(kv_source)[..., D:]

    @property
    def q_weight(self) -> torch.Tensor:
        """Slice of the (possibly parametrised) projection weight that produces Q."""
        D = self.embed_dim
        if self.attn_type == "self":
            return self.qkv_proj.weight[:D, :]
        return self.q_proj.weight

    @property
    def k_weight(self) -> torch.Tensor:
        """Slice of the (possibly parametrised) projection weight that produces K."""
        D = self.embed_dim
        if self.attn_type == "self":
            return self.qkv_proj.weight[D : 2 * D, :]
        return self.kv_proj.weight[:D, :]

    @property
    def v_weight(self) -> torch.Tensor:
        """Slice of the (possibly parametrised) projection weight that produces V."""
        D = self.embed_dim
        if self.attn_type == "self":
            return self.qkv_proj.weight[2 * D :, :]
        return self.kv_proj.weight[D:, :]

    def forward(
        self,
        q_source: torch.Tensor,
        kv_source: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_source:  ``[B, T_q, D]``  — what asks (Query).
            kv_source: ``[B, T_kv, D]`` — what is read (Key, Value).
                For self-attention ``kv_source`` should be the same tensor
                as ``q_source``; only ``q_source`` is fed through the fused
                ``qkv_proj``.
            attn_mask: optional boolean mask ``[T_q, T_kv]`` where ``True``
                means "do not attend". Entries marked ``True`` get ``-inf``
                added before softmax. Cross-attention supplies the windowed
                mask; self-attention passes ``None``.

        Returns:
            ``[B, T_q, D]``
        """
        B, T_q, D = q_source.shape
        T_kv = kv_source.shape[1]
        h, d_h = self.n_heads, self.d_head

        if self.attn_type == "self":
            qkv = self.qkv_proj(q_source)                              # [B, T_q, 3D]
            Q, K, V = qkv.split(D, dim=-1)
        else:
            Q = self.q_proj(q_source)                                  # [B, T_q,  D]
            kv = self.kv_proj(kv_source)                               # [B, T_kv, 2D]
            K, V = kv.split(D, dim=-1)

        Q = Q.view(B, T_q, h, d_h).transpose(1, 2)                     # [B, h, T_q,  d_h]
        K = K.view(B, T_kv, h, d_h).transpose(1, 2)                    # [B, h, T_kv, d_h]
        V = V.view(B, T_kv, h, d_h).transpose(1, 2)                    # [B, h, T_kv, d_h]

        if self.qk_norm:
            Q = self.q_ln(Q)
            K = self.k_ln(K)

        attn = (Q @ K.transpose(-2, -1)) / self.scale                  # [B, h, T_q, T_kv]

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        if self.capture_attn:
            self._last_attn = attn.detach()
        attn = self.attn_drop(attn)

        out = attn @ V                                                 # [B, h, T_q, d_h]
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)         # [B, T_q, D]
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    """Position-wise FFN: ``Linear(D → 4D) → GELU → Dropout → Linear(4D → D) → Dropout``."""

    def __init__(self, embed_dim: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = embed_dim * ffn_ratio
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    """Per-channel learnable gain applied to a sub-layer output before the residual add.

    Introduced in CaiT (Touvron et al. 2021) and used in DeiT-III, DINOv2, DINOv3.
    The gain ``γ`` is initialised close to zero, so at step 0 each sub-block is
    effectively an identity — the residual dominates. Training then learns how
    much each sub-block should contribute.

    Usage inside a sub-block::

        x = x + LayerScale(SubLayer(LayerNorm(x)))

    This module is a no-op in parameter count when replaced by ``nn.Identity()``.
    """

    def __init__(self, embed_dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


def _make_layer_scale(embed_dim: int, init_value: Optional[float]) -> nn.Module:
    """Return a ``LayerScale(embed_dim)`` if ``init_value`` is a float, else ``Identity``.

    Keeping the off-path as ``nn.Identity`` means LayerScale is **completely absent**
    from the parameter list and forward graph when disabled — no stored ``γ``, no
    extra multiply. Makes the ablation a clean toggle rather than a learned-near-one
    approximation.
    """
    if init_value is None:
        return nn.Identity()
    return LayerScale(embed_dim, init_value=init_value)


def build_2d_sincos_pe(
    h: int,
    w: int,
    embed_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    temperature: Optional[float] = None,
) -> torch.Tensor:
    """Fixed 2D sinusoidal positional encoding of shape ``[h*w, embed_dim]``.

    First half of the feature dim encodes the y coordinate, second half encodes x.
    Parameter-free so it adapts to any input resolution.

    ``temperature`` controls the spread of the frequency bands; setting it to
    ``T`` makes the slowest-frequency wavelength ``2π·T`` and the fastest
    ``2π``. The standard Transformer/ViT convention of ``T=10000`` was tuned
    for sequences of length ~thousands, but on small image-patch grids
    (e.g. 16x16 = 256 positions) almost every frequency band is effectively
    DC and the resulting PE has pairwise off-diag cosine ≥ 0.65 between
    every pair of positions — so the PE provides essentially a constant
    spatial bias rather than a discriminative coordinate frame. When
    ``temperature is None`` (the default) we instead pick a grid-aware
    temperature ``T = max(max(h, w), 2)`` so the band actually covers the
    discriminable range of the grid: distant positions are near-orthogonal
    (off-diag cosine ~ -0.1) while axis-adjacent positions are highly
    similar (~ +0.96) — the expected locality property of a 2-D PE. Pass
    an explicit ``temperature`` if you need to reproduce the legacy
    convention.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sin-cos PE")

    if temperature is None:
        temperature = float(max(max(h, w), 2))

    d_axis = embed_dim // 2               # dims per axis
    n_freq = d_axis // 2                  # sin/cos pairs per axis
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    # frequency bands: div[i] = 1 / T^(i/n_freq)
    div = torch.exp(
        torch.arange(n_freq, device=device, dtype=dtype) * (-math.log(temperature) / n_freq)
    )  # [n_freq]

    y_pos = torch.arange(h, device=device, dtype=dtype)  # [h]
    x_pos = torch.arange(w, device=device, dtype=dtype)  # [w]

    # per-axis embedding: [length, d_axis] = concat(sin, cos) for each freq
    def _axis_emb(pos: torch.Tensor) -> torch.Tensor:
        e = torch.zeros(pos.shape[0], d_axis, device=device, dtype=dtype)
        e[:, 0::2] = torch.sin(pos.unsqueeze(-1) * div)
        e[:, 1::2] = torch.cos(pos.unsqueeze(-1) * div)
        return e

    y_emb = _axis_emb(y_pos)              # [h, d_axis]
    x_emb = _axis_emb(x_pos)              # [w, d_axis]

    y_grid = y_emb.unsqueeze(1).expand(h, w, d_axis)    # [h, w, d_axis]
    x_grid = x_emb.unsqueeze(0).expand(h, w, d_axis)    # [h, w, d_axis]
    pe = torch.cat([y_grid, x_grid], dim=-1)            # [h, w, D]
    return pe.reshape(h * w, embed_dim)                 # [P, D]


def _make_slot_embed_init(
    num_slots: int,
    embed_dim: int,
    *,
    mode: str = "random",
    std: float = 0.02,
) -> torch.Tensor:
    """Initialiser for the prior bank's learnable slot identity ``[M, d]``.

    Modes:

    * ``"random"`` — i.i.d. Gaussian with std ``std``. Statistically
      equivalent to enlarging ``prior_init_std``; does **not** break
      slot-permutation symmetry on its own. Kept for backward
      compatibility with the early slot_embed experiments.
    * ``"sincos_2d"`` — 2-D sin/cos PE on a ``√M × √M`` grid via
      :func:`build_2d_sincos_pe`. Each slot gets a deterministic
      ``(row, col)``-indexed identity that mirrors the CHM token
      grid's own PE, giving a stable spatial coordinate frame for
      the bank. Adjacent slots have high cosine, distant slots have
      low/negative cosine — exactly the structure ``"random"`` lacks.
      Requires ``num_slots`` to be a perfect square and
      ``embed_dim % 4 == 0``.
    """
    if mode == "random":
        return torch.randn(num_slots, embed_dim) * std
    if mode == "sincos_2d":
        side = int(round(math.sqrt(num_slots)))
        if side * side != num_slots:
            raise ValueError(
                f"slot_embed_init='sincos_2d' requires num_prior_tokens to be a "
                f"perfect square, got {num_slots} (nearest squares: "
                f"{side * side}, {(side + 1) ** 2})."
            )
        if embed_dim % 4 != 0:
            raise ValueError(
                f"slot_embed_init='sincos_2d' requires embed_dim % 4 == 0, "
                f"got embed_dim={embed_dim}."
            )
        # Grid-aware temperature is now the default of build_2d_sincos_pe
        # (T = max(max(h, w), 2)), so we do not need to override it here.
        # The default lands at T=side for our square grids and gives
        # off-diag cosine ranging from near-0 (distant slots) to ~+0.96
        # (axis-adjacent slots) — the locality property a 2-D PE should
        # have. See the docstring of build_2d_sincos_pe for why the
        # legacy T=10000 default was wrong on small image-patch grids.
        return build_2d_sincos_pe(side, side, embed_dim).contiguous()
    raise ValueError(
        f"slot_embed_init must be 'random' or 'sincos_2d', got {mode!r}."
    )


# ---------------------------------------------------------------------------
# Windowed cross-attention mask helpers
# ---------------------------------------------------------------------------
#
# Hoisted to module level so every model in this file (``Dinov3HeightModel``,
# ``Dinov3HeightModelDPT``) can share the exact same locality-prior logic and
# cache entries without duplicating methods.


def build_window_mask(h: int, w: int, k: int, device: torch.device) -> torch.Tensor:
    """``[P, P]`` bool mask where ``True`` means "mask out".

    A query at grid position ``(y_q, x_q)`` attends to key ``(y_k, x_k)`` iff
    ``|y_q - y_k| <= k`` and ``|x_q - x_k| <= k``. Assumes the image query grid
    and the CHM key grid have the same shape ``[h, w]`` (true for the current
    ``CHMPromptEncoder`` whose stride matches the backbone patch size, and for
    the zero-memory fallback, which allocates ``h*w`` zero tokens).
    """
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    pos = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # [P, 2]
    dy = (pos[:, None, 0] - pos[None, :, 0]).abs()           # [P, P]
    dx = (pos[:, None, 1] - pos[None, :, 1]).abs()
    return (dy > k) | (dx > k)                               # [P, P] bool


def get_cached_window_mask(
    cache: dict, h: int, w: int, k: int, device: torch.device
) -> torch.Tensor:
    """Return a cached ``[P, P]`` bool window mask for the given grid + device.

    ``cache`` is any dict-like; the caller owns it so each model instance keeps
    its own cache and we avoid cross-instance leakage on GPU switching.
    """
    key = (
        h, w, k,
        device.type,
        device.index if device.index is not None else -1,
    )
    mask = cache.get(key)
    if mask is None:
        mask = build_window_mask(h, w, k, device)
        cache[key] = mask
    return mask


# ---------------------------------------------------------------------------
# Sub-blocks (each one owns its Pre-LN + residual)
# ---------------------------------------------------------------------------


class CrossAttnSubBlock(nn.Module):
    """Cross-attention sub-block.

    Implements::

        x = x + LayerScale( MHA( LN(x),  LN(memory), attn_mask ) )

    Residual is internal — caller does not need to add anything. ``attn_mask``
    encodes the *locality prior*: entries marked ``True`` get ``-inf`` added
    before softmax, so a query only attends to memory tokens inside an allowed
    window (see ``Dinov3HeightModel._build_window_mask``).

    LayerScale is optional (controlled by ``layer_scale_init``). When disabled
    it is replaced by ``nn.Identity`` so no parameters or extra compute exist.

    The MHA knobs (``qk_norm``, ``qkv_spectral_norm``,
    ``qkv_spectral_learnable``) are forwarded to :class:`MultiHeadAttention`.
    See its docstring for exact semantics. All default to no-op.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        layer_scale_init: Optional[float] = None,
        qk_norm: bool = False,
        qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(
            embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_type="cross",
            qk_norm=qk_norm,
            qkv_spectral_norm=qkv_spectral_norm,
            qkv_spectral_learnable=qkv_spectral_learnable,
            qkv_spectral_gamma_init=qkv_spectral_gamma_init,
        )
        self.ls = _make_layer_scale(embed_dim, layer_scale_init)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.ls(
            self.mha(self.norm_q(x), self.norm_kv(memory), attn_mask=attn_mask)
        )


class SelfAttnSubBlock(nn.Module):
    """Self-attention sub-block.

    Implements::

        x = x + LayerScale( MHA( LN(x), LN(x) ) )

    MHA knobs (``qk_norm``, ``qkv_spectral_norm``, ``qkv_spectral_learnable``)
    are forwarded to :class:`MultiHeadAttention`. All default to no-op.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        layer_scale_init: Optional[float] = None,
        qk_norm: bool = False,
        qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(
            embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_type="self",
            qk_norm=qk_norm,
            qkv_spectral_norm=qkv_spectral_norm,
            qkv_spectral_learnable=qkv_spectral_learnable,
            qkv_spectral_gamma_init=qkv_spectral_gamma_init,
        )
        self.ls = _make_layer_scale(embed_dim, layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ln = self.norm(x)
        return x + self.ls(self.mha(x_ln, x_ln))


class FFNSubBlock(nn.Module):
    """Feed-forward sub-block.

    Implements::

        x = x + LayerScale( FFN( LN(x) ) )
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        layer_scale_init: Optional[float] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_ratio=ffn_ratio, dropout=dropout)
        self.ls = _make_layer_scale(embed_dim, layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ls(self.ffn(self.norm(x)))


# ---------------------------------------------------------------------------
# Decoder layer (3 sub-blocks) and stack (N layers)
# ---------------------------------------------------------------------------


class HeightDecoderLayer(nn.Module):
    """One Transformer decoder layer: ``CrossAttn → SelfAttn → FFN``.

    Each sub-block is Pre-LN + residual, so ``x`` is refined three times
    per layer and never replaced.

    The MHA stability knobs (``qk_norm``, ``qkv_spectral_norm``) can be
    set independently on the cross- and self-attention sub-blocks via
    ``cross_attn_*`` and ``self_attn_*`` kwargs. Defaults are no-ops, which
    yields a vanilla pre-LN decoder layer.

    LayerScale init can also be set asymmetrically: ``layer_scale_init``
    sets the default for all three sub-blocks, and
    ``cross_attn_layer_scale_init`` overrides just the cross-attn path.
    Use this to give the cross-attn residual a higher initial gain (e.g.
    1e-1 vs 1e-5 on the other paths) so the optimiser doesn't suppress
    the CHM-conditional pathway in early training.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        layer_scale_init: Optional[float] = None,
        cross_attn_layer_scale_init: Optional[float] = None,
        cross_attn_qk_norm: bool = False,
        cross_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        self_attn_qk_norm: bool = False,
        self_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
    ):
        super().__init__()
        # cross_attn_layer_scale_init=None → inherit layer_scale_init.
        # cross_attn_layer_scale_init=<float> → use that for cross-attn only.
        cross_ls_init = (
            cross_attn_layer_scale_init
            if cross_attn_layer_scale_init is not None
            else layer_scale_init
        )
        self.cross_attn = CrossAttnSubBlock(
            embed_dim, n_heads=n_heads, dropout=dropout,
            layer_scale_init=cross_ls_init,
            qk_norm=cross_attn_qk_norm,
            qkv_spectral_norm=cross_attn_qkv_spectral_norm,
            qkv_spectral_learnable=qkv_spectral_learnable,
            qkv_spectral_gamma_init=qkv_spectral_gamma_init,
        )
        self.self_attn = SelfAttnSubBlock(
            embed_dim, n_heads=n_heads, dropout=dropout,
            layer_scale_init=layer_scale_init,
            qk_norm=self_attn_qk_norm,
            qkv_spectral_norm=self_attn_qkv_spectral_norm,
            qkv_spectral_learnable=qkv_spectral_learnable,
            qkv_spectral_gamma_init=qkv_spectral_gamma_init,
        )
        self.ffn = FFNSubBlock(
            embed_dim, ffn_ratio=ffn_ratio, dropout=dropout,
            layer_scale_init=layer_scale_init,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        return_intermediates: bool = False,
    ):
        """
        Args:
            x:      ``[B, P, D]`` image queries being refined
            memory: ``[B, P', D]`` CHM tokens with PE (K/V source for cross-attn)
            cross_attn_mask: ``[P, P']`` bool mask for the cross-attention;
                ``True`` means "do not attend". Self-attention stays global —
                image patches must be able to exchange context beyond the window.

        Returns:
            If ``return_intermediates=False``: ``[B, P, D]``.
            Else: ``(output, dict)`` where the dict contains per-sub-block outputs.
        """
        x_xa = self.cross_attn(x, memory, attn_mask=cross_attn_mask)
        x_sa = self.self_attn(x_xa)
        x_ffn = self.ffn(x_sa)

        if return_intermediates:
            # ``cross_attn_delta`` = ``x_xa - x`` = the residual contribution
            # of the cross-attn sub-block (= ``LayerScale(MHA(LN(x), LN(memory)))``
            # before the residual add). Needed by the per-layer CHM-prediction
            # probe so the linear head sees only the CHM-conditional injection,
            # not the full residual stream which carries image-only info.
            return x_ffn, {
                "before_cross_attn": x,
                "after_cross_attn": x_xa,
                "cross_attn_delta": x_xa - x,
                "after_self_attn": x_sa,
                "after_ffn": x_ffn,
            }
        return x_ffn


class HeightDecoderStack(nn.Module):
    """Stack of ``n_layers`` identical ``HeightDecoderLayer`` modules, + final LayerNorm.

    The ``cross_attn_*`` and ``self_attn_*`` kwargs configure the MHA
    stability knobs (QK-Norm / joint-QKV σReparam); every layer in the
    stack receives the same setting (they share the same mechanism, not
    the same weights). Defaults are no-ops, so this is a vanilla pre-LN
    decoder stack out of the box.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        layer_scale_init: Optional[float] = None,
        cross_attn_layer_scale_init: Optional[float] = None,
        cross_attn_qk_norm: bool = False,
        cross_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        self_attn_qk_norm: bool = False,
        self_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            HeightDecoderLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                layer_scale_init=layer_scale_init,
                cross_attn_layer_scale_init=cross_attn_layer_scale_init,
                cross_attn_qk_norm=cross_attn_qk_norm,
                cross_attn_qkv_spectral_norm=cross_attn_qkv_spectral_norm,
                self_attn_qk_norm=self_attn_qk_norm,
                self_attn_qkv_spectral_norm=self_attn_qkv_spectral_norm,
                qkv_spectral_learnable=qkv_spectral_learnable,
                qkv_spectral_gamma_init=qkv_spectral_gamma_init,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        return_intermediates: bool = False,
    ):
        """
        Args:
            x:      ``[B, P, D]`` starting image tokens
            memory: ``[B, P', D]`` CHM tokens with PE
            cross_attn_mask: ``[P, P']`` bool mask (``True`` = mask) applied to
                the cross-attention of every layer. Built once per image size
                by ``Dinov3HeightModel._build_window_mask``.

        Returns:
            If ``return_intermediates=False``: ``[B, P, D]`` refined tokens.
            Else: ``(output, list_of_dicts)`` where list is one entry per layer.
        """
        per_layer = []
        for layer in self.layers:
            if return_intermediates:
                x, inter = layer(
                    x, memory, cross_attn_mask,
                    return_intermediates=True,
                )
                per_layer.append(inter)
            else:
                x = layer(x, memory, cross_attn_mask)

        x = self.final_norm(x)

        if return_intermediates:
            return x, per_layer
        return x


# ---------------------------------------------------------------------------
# CHM prompt encoder (CNN → tokens) and Conv head (tokens → height map)
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for ``[B, C, H, W]`` feature maps.

    Equivalent to ``nn.LayerNorm(C)`` applied at every spatial location —
    statistics are computed per-sample, per-position over the channel dim.
    Removes the train/eval running-statistic coupling that ``BatchNorm2d``
    introduces, which is important for the CHM encoder because the input
    distribution shifts heavily between train (noisy, corrupted CHM) and
    eval (cleaner CHM).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] → [B, H, W, C] → LN(C) → [B, C, H, W]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class CHMPromptEncoder(nn.Module):
    """Four-stage CNN that encodes a single-channel CHM to patch-level tokens.

    Strides match the ViT patch grid (2×2×2×2 = 16×) so output tokens line up
    spatially with backbone features at H/16 × W/16. Deeper than a two-conv
    stem to give the encoder more capacity to learn "LiDAR gives absolute
    scale, ignore the noise" rather than memorising a shallow distribution.

    Uses :class:`LayerNorm2d` so the encoder is not coupled to the training
    CHM distribution via BN running statistics — matters because corrupted
    train CHMs and cleaner deploy CHMs are statistically different.

    Channel schedule (stride ×2 per stage, total ×16):

    =====  ===============  =========
    Stage   Channels        Output
    =====  ===============  =========
      1        1 →    32     H/2 × W/2
      2       32 →    64     H/4 × W/4
      3       64 →   256     H/8 × W/8
      4      256 →  embed    H/16 × W/16
    =====  ===============  =========
    """

    def __init__(self, embed_dim: int = 1024, patch_size: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, chm: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            chm: ``[B, 1, H, W]`` raw CHM values in metres.

        Returns:
            tokens:  ``[B, P', D]`` with ``P' = h' * w'``
            h', w':  spatial dims of the encoded CHM feature map
        """
        x = self.encoder(chm)                             # [B, D, h', w']
        B, D, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)             # [B, P', D]
        return tokens, h, w


class CHMReconHead(nn.Module):
    """Auxiliary decoder: CHM token grid → reconstructed input CHM map.

    Mirrors :class:`CHMPromptEncoder`'s 4-stage channel ladder in reverse
    (1024 → 256 → 64 → 32 → 1) with bilinear upsampling before each
    ``3×3`` conv, giving a total ×16 upscaling back to image resolution.

    The head is attached off the CHM encoder's output tokens and trained
    against the **(corrupted) input CHM** — this is an autoencoder-style
    objective on the CHM stream: the encoder must keep enough information
    in its tokens to reproduce its own input, which guards against
    representation collapse and against the encoder silently leaking clean
    height info into the cross-attention. The *final* nDSM prediction is
    still supervised against the clean GT in the main loss; see the Plan v1
    ablation-study page for the full rationale.

    Used only when ``aux_chm_recon=True`` on the parent model and
    ``aux_chm_recon_weight > 0`` in the training config; otherwise the head
    isn't instantiated at all.
    """

    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            LayerNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            LayerNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, tokens_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens_grid: ``[B, D, h, w]`` CHM tokens laid out spatially
                (output of :class:`CHMPromptEncoder`'s 2D feature map,
                pre-flatten).

        Returns:
            ``[B, 1, H, W]`` reconstructed CHM at image resolution
            (``H = 16 * h``, ``W = 16 * w``). Trained against the
            (corrupted) input CHM, not the clean GT nDSM.
        """
        return self.decoder(tokens_grid)


class SimpleConvHead(nn.Module):
    """Conv decoder: project → 4× ConvTranspose upsampling → 1×1 output.

    Starts at backbone resolution (H/16 × W/16), upsamples 16× back to (H, W).
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),  # ×2
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),          # ×2
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),           # ×2
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),            # ×2
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Args:
            x: ``[B, C, h, w]`` feature map at backbone resolution.
        """
        x = self.decoder(x)
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
        return x


# ---------------------------------------------------------------------------
# Plan v2: Learnable LiDAR prior bank + posterior updater
# ---------------------------------------------------------------------------
#
# Plan v2 replaces the degenerate "zero memory" fallback of Plan v1 with a
# small learnable bank of prior tokens that encodes general height knowledge
# distilled from training. When a CHM is provided, a single cross-attention
# "posterior updater" refreshes the prior against the observed evidence
# before the image decoder sees it; when no CHM is provided, the prior alone
# acts as the memory.
#
# The building blocks below are intentionally additive — pure composition of
# `MultiHeadAttention` + `LayerNorm` + an `nn.Parameter` bank. They are
# consumed by ``Dinov3HeightModelV2`` / ``Dinov3HeightModelV2DPT`` further
# down this file. The Plan v1 classes (``Dinov3HeightModel`` /
# ``Dinov3HeightModelDPT``) are *not* modified, so existing checkpoints
# trained against Plan v1 continue to load unchanged.
#
# See ``docs/planning-site/src/content/docs/treeheight/lidar-prior.mdx`` for
# the architectural rationale, the terminology (prior/evidence/posterior),
# and the Track A diagnostics (P1–P5) that read signals off these modules.


class PosteriorUpdater(nn.Module):
    """One cross-attention layer: prior queries CHM evidence, *gated* residual update.

    All five gate variants share the same pre-normalised cross-attention step::

        q  = LN(prior)
        kv = LN(chm_tokens)
        update = MHA(q, kv, kv)

    and then differ only in how ``update`` is combined with ``prior``:

    =================  =======================================================
    ``gate`` value     Residual form
    =================  =======================================================
    ``post_ln_alpha``  ``posterior = prior + α · LN(update)``   *(legacy)*
    ``tanh``           ``posterior = prior + tanh(α) · update`` *(Flamingo)*
    ``layerscale``     ``posterior = prior + γ ⊙ update``       *(CaiT/DINOv3)*
    ``gru``            ``posterior = GRUCell(update, prior)``   *(Slot Attn.)*
    ``alpha``          ``posterior = prior + α · update``       *(U4 saliency)*
    =================  =======================================================

    The ``alpha`` gate (added for U4) is the bare-α variant of
    ``post_ln_alpha``: same scalar learnable α (``nn.Parameter``), but
    with **no** post-attention LayerNorm and **no** tanh wrapper. It
    preserves the magnitude *of every individual update token*
    (saliency-weighted writes — high-confidence query tokens write
    more, low-confidence ones write almost nothing) so the prior bank
    can be incrementally refined rather than being uniformly
    overwritten. See ``docs/chm/insights/U_series_stabilization_plan.md``
    §3.4 (memory-bank preservation).

    ``post_ln_alpha`` preserves the pre-2026-04 parameter layout exactly
    (``norm_update`` LayerNorm + scalar ``alpha``) so historical checkpoints
    — including the K0/K5/T0 runs trained under ``normalize_update=True`` +
    ``update_scale_init=<float>`` — load bit-for-bit under this gate.

    The other three gates drop the post-attention LN and replace the simple
    residual add with a gated form designed so the updater starts as a
    near-no-op and only opens up during training if useful:

    * ``tanh``: single learnable scalar ``α`` (init 0 → ``tanh(0)=0`` gives a
      *true* no-op at step 0; bounded ≤ 1 at steady state).
    * ``layerscale``: per-channel learnable gain ``γ`` (init 1e-5; same as
      the DINOv3 backbone's own LayerScale, unbounded in training).
    * ``gru``: ``nn.GRUCell`` acting on ``(update, prior)``; the update-gate
      bias is initialised **positive** so that ``z ≈ 1`` in PyTorch's
      ``h_new = (1-z)·n + z·h_prev``, keeping the initial blend mostly
      ``prior``.

    See ``docs/chm/insights/paradigm_shift_ln_removal.md`` for the rationale
    behind the four options and when each is expected to win.
    """

    VALID_GATES = ("post_ln_alpha", "tanh", "layerscale", "gru", "alpha")

    # Defaults chosen so each gate starts as a no-op (or as close as the
    # mechanism allows). Callers are encouraged to set ``gate_init`` explicitly
    # anyway; these are safety nets for the default config.
    _GATE_DEFAULT_INIT = {
        "post_ln_alpha": 0.1,    # historic K0 value, NOT a no-op; must override
        "tanh": 0.0,             # true no-op: tanh(0) = 0
        "layerscale": 1e-5,      # near no-op; DINOv3 backbone convention
        "gru": 3.0,              # update-gate bias (positive) => z ≈ sigmoid(+6) ≈ 0.998,
                                  # so h_new = (1-z)·n + z·h_prev ≈ h_prev (= prior). See
                                  # _init_gru_update_gate_bias for the derivation.
        "alpha": 0.03,           # bare-α (U4): prior + α·update. Matches T11's α
                                  # (post_ln_alpha gate_init=0.03) so per-iter
                                  # write magnitude is comparable.
    }

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        gate: Optional[str] = None,
        gate_init: Optional[float] = None,
        qk_norm: bool = False,
        qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
        # --- Deprecated kwargs (still accepted for backward compatibility) ---
        normalize_update: Optional[bool] = None,
        update_scale_init: Optional[float] = None,
    ):
        super().__init__()

        gate, gate_init = self._resolve_gate_spec(
            gate=gate,
            gate_init=gate_init,
            normalize_update=normalize_update,
            update_scale_init=update_scale_init,
        )
        if gate not in self.VALID_GATES:
            raise ValueError(
                f"gate must be one of {self.VALID_GATES}, got {gate!r}"
            )
        self.gate_type = gate
        self.gate_init_value = float(gate_init)

        # Pre-normalisation on Q and KV — standard modern cross-attention
        # convention (Perceiver IO, Flamingo, DINOv3 backbone, etc.). Retained
        # for every gate.
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(
            embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_type="cross",
            qk_norm=qk_norm,
            qkv_spectral_norm=qkv_spectral_norm,
            qkv_spectral_learnable=qkv_spectral_learnable,
            qkv_spectral_gamma_init=qkv_spectral_gamma_init,
        )

        if gate == "post_ln_alpha":
            # LEGACY state_dict layout. `norm_update` is a real LayerNorm and
            # `alpha` is a scalar Parameter, matching the pre-rewrite module
            # exactly so old checkpoints (K0, K5, T0) load without remapping.
            self.norm_update = nn.LayerNorm(embed_dim)
            self.alpha = nn.Parameter(torch.tensor(self.gate_init_value))
        elif gate == "tanh":
            self.alpha = nn.Parameter(torch.tensor(self.gate_init_value))
        elif gate == "alpha":
            # U4 bare-α gate: prior + α·update.
            # No post-attention LN (preserves per-token update magnitudes
            # so saliency-weighted writes are possible), no tanh wrapper
            # (no built-in cap — pairs with σReparam upstream which already
            # bounds the spectral norm of every linear, so update magnitude
            # is bounded by data + γ rather than by a hard 1.0 cap).
            self.alpha = nn.Parameter(torch.tensor(self.gate_init_value))
        elif gate == "layerscale":
            self.gamma = nn.Parameter(
                torch.full((embed_dim,), self.gate_init_value)
            )
        elif gate == "gru":
            self.gru = nn.GRUCell(embed_dim, embed_dim)
            self._init_gru_update_gate_bias(self.gru, self.gate_init_value)

    # ------------------------------------------------------------------
    # Gate-spec resolver: translate old (normalize_update, update_scale_init)
    # kwargs to the new (gate, gate_init) pair. Exactly one API may be used.
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_gate_spec(
        gate: Optional[str],
        gate_init: Optional[float],
        normalize_update: Optional[bool],
        update_scale_init: Optional[float],
    ):
        old_spec_used = (normalize_update is not None) or (update_scale_init is not None)
        new_spec_used = (gate is not None) or (gate_init is not None)

        if old_spec_used and new_spec_used:
            raise ValueError(
                "PosteriorUpdater received both the new (gate, gate_init) API "
                "and the deprecated (normalize_update, update_scale_init) API. "
                "Use exactly one — prefer the new one in fresh configs."
            )

        if old_spec_used:
            # The only combination that appeared in shipped configs is
            # normalize_update=True + update_scale_init=<float>. Anything
            # else either never ran or was never intended, so fail loudly.
            if normalize_update and update_scale_init is not None:
                return "post_ln_alpha", float(update_scale_init)
            raise ValueError(
                "Deprecated PosteriorUpdater kwargs must come as "
                "normalize_update=True + update_scale_init=<float> (the "
                "legacy post_ln_alpha contract). Got "
                f"normalize_update={normalize_update}, "
                f"update_scale_init={update_scale_init}. Switch to the "
                "explicit (gate, gate_init) API instead."
            )

        # New API path (or neither specified — use backward-compatible default).
        if gate is None:
            gate = "post_ln_alpha"
        if gate_init is None:
            gate_init = PosteriorUpdater._GATE_DEFAULT_INIT[gate]
        return gate, float(gate_init)

    @staticmethod
    def _init_gru_update_gate_bias(cell: nn.GRUCell, bias_init: float) -> None:
        """Seed the GRU's update-gate bias so the initial blend stays on ``prior``.

        PyTorch's ``nn.GRUCell`` computes, for each forward step::

            z     = sigmoid(W_iz·x + b_iz + W_hz·h + b_hz)   # update gate
            h_new = (1 - z) · n + z · h                       # blend with candidate ``n``

        so ``posterior ≈ prior`` requires ``z ≈ 1``, which needs a *positive*
        bias. ``nn.GRUCell`` concatenates biases for the three gates
        (reset, update, new) in both ``bias_ih`` and ``bias_hh``; writing
        ``bias_init`` into *both* means the effective pre-activation for ``z``
        starts at ``2·bias_init``. With ``bias_init = 3.0`` the effective
        logit is 6, so ``sigmoid(6) ≈ 0.998`` and ``posterior ≈ 0.998 · prior
        + 0.002 · n ≈ prior`` at step 0 — the "near-no-op" property the
        other gates also have at init.
        """
        with torch.no_grad():
            hidden_size = cell.hidden_size
            for bias in (cell.bias_ih, cell.bias_hh):
                # bias layout per PyTorch docs: [reset | update | new].
                # Only the update-gate slice is touched; reset/new stay at
                # PyTorch's default uniform init.
                bias[hidden_size : 2 * hidden_size].fill_(bias_init)

    # ------------------------------------------------------------------
    def forward(
        self,
        prior: torch.Tensor,
        chm_tokens: torch.Tensor,
        slot_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gated residual update of the prior from CHM evidence.

        Args:
            prior:      ``[B, M, d]`` prior slots (queries).
            chm_tokens: ``[B, P', d]`` CHM evidence (keys/values).
            slot_embed: optional ``[M, d]`` or ``[B, M, d]`` learnable
                slot-identity embedding (DETR ``query_embed`` pattern).
                When provided, it is added to ``prior`` *only* on the Q
                path — the residual update is still applied to bare
                ``prior``, so ``slot_embed`` flavours every iteration's
                queries with a persistent slot identity without
                accumulating in the posterior.

        Returns:
            ``[B, M, d]`` posterior tokens, same shape as ``prior``.
        """
        if slot_embed is not None:
            if slot_embed.dim() == 2:
                slot_embed = slot_embed.unsqueeze(0)
            q_input = prior + slot_embed
        else:
            q_input = prior
        q = self.norm_q(q_input)
        kv = self.norm_kv(chm_tokens)
        update = self.cross_attn(q, kv)

        if self.gate_type == "post_ln_alpha":
            return prior + self.alpha * self.norm_update(update)
        if self.gate_type == "tanh":
            return prior + torch.tanh(self.alpha) * update
        if self.gate_type == "alpha":
            return prior + self.alpha * update
        if self.gate_type == "layerscale":
            return prior + self.gamma * update
        if self.gate_type == "gru":
            B, M, d = prior.shape
            posterior = self.gru(
                update.reshape(B * M, d),
                prior.reshape(B * M, d),
            )
            return posterior.reshape(B, M, d)
        raise AssertionError(f"unreachable: gate_type={self.gate_type!r}")


class LidarPriorLayer(nn.Module):
    """Learnable LiDAR prior bank with optional posterior update.

    Owns a single ``nn.Parameter`` prior bank ``prior ∈ R^[M, d]`` and a
    :class:`PosteriorUpdater`. Produces the memory tensor that the image
    decoder's cross-attention consumes — handling both the CHM-absent
    (prior-only) and CHM-present (posterior-updated) regimes in one call.

    Three concrete output shapes::

        chm_tokens is None                      →  [B, M, d]         (prior only)
        chm_tokens is [B, P', d], concat=False  →  [B, M, d]         (pure B, default)
        chm_tokens is [B, P', d], concat=True   →  [B, M + P', d]    (A+B hybrid)

    The concat path is an opt-in diagnostic / early-training aid — see the
    planning site page for why pure B is the default (information bottleneck
    forces the updater to be genuinely useful).

    Args:
        embed_dim:            token dimensionality ``d``.
        num_prior_tokens:     size of the prior bank ``M``. Default 64.
        n_heads:              heads in the posterior updater's MHA. Default 8.
        dropout:              dropout inside the updater's MHA.
        concat_chm_to_memory: if True, append raw CHM tokens after the
            posterior. Default False (pure Option B from the plan).
        prior_init_std:       std for the prior bank initialization.
            Default 0.02. Larger values (0.1–0.5) give slots more initial
            diversity and closer scale to the CHM encoder output.
        slot_embed:           if True, instantiate a learnable
            ``[M, d]`` slot-identity embedding (DETR ``query_embed``
            pattern) that is added to ``prior`` on the updater's Q path
            only. Combined with a near-zero ``prior_init_std`` and a
            structured ``slot_embed_init`` (sincos_2d) it gives every
            slot a *deterministic* identity that gradients can stably
            attribute to a fixed slot index — the necessary first
            condition for slot specialisation. The residual update
            still uses bare ``prior``, so the embedding only flavours
            every Q without polluting the running posterior.
        slot_embed_init:      ``"random"`` (default; i.i.d. Gaussian,
            statistically equivalent to enlarging ``prior_init_std`` —
            does **not** break slot-permutation symmetry on its own) or
            ``"sincos_2d"`` (2-D sin/cos PE indexed by ``(row, col)`` on
            a ``√M × √M`` grid; reuses ``build_2d_sincos_pe``). The
            sincos variant requires ``M`` to be a perfect square and
            ``embed_dim`` divisible by 4. It produces structured,
            deterministic per-slot identities — adjacent slots have
            high cosine, distant slots low/negative cosine — giving
            the network a stable spatial coordinate frame for the
            bank that mirrors the CHM token grid.
        slot_embed_init_std:  std for the slot embedding's i.i.d. Gaussian
            init. Used only when ``slot_embed_init="random"``.
        gate:                 residual gate for the posterior update. One of
            ``"post_ln_alpha"`` (legacy; LN + scalar α), ``"tanh"``,
            ``"layerscale"``, ``"gru"``. See :class:`PosteriorUpdater`.
        gate_init:            init value for the chosen gate (α for
            ``post_ln_alpha``/``tanh``, per-channel γ for ``layerscale``,
            update-gate bias for ``gru``). Defaults come from
            :attr:`PosteriorUpdater._GATE_DEFAULT_INIT`.
        normalize_update:     DEPRECATED — old v1 of the API. Pass
            ``gate="post_ln_alpha"`` instead.
        update_scale_init:    DEPRECATED — old v1 of the API. Pass
            ``gate_init=<float>`` instead. When combined with
            ``normalize_update=True`` it is translated into the new API
            automatically so checkpoints from the pre-rewrite era load.
    """

    def __init__(
        self,
        embed_dim: int,
        num_prior_tokens: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
        concat_chm_to_memory: bool = False,
        prior_init_std: float = 0.02,
        slot_embed: bool = False,
        slot_embed_init: str = "random",
        slot_embed_init_std: float = 0.02,
        gate: Optional[str] = None,
        gate_init: Optional[float] = None,
        qk_norm: bool = False,
        qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
        # --- Deprecated kwargs (accepted + translated for backward compat) ---
        normalize_update: Optional[bool] = None,
        update_scale_init: Optional[float] = None,
    ):
        super().__init__()
        if num_prior_tokens <= 0:
            raise ValueError(
                f"num_prior_tokens must be positive, got {num_prior_tokens}"
            )
        self.embed_dim = embed_dim
        self.num_prior_tokens = num_prior_tokens
        self.concat_chm_to_memory = concat_chm_to_memory

        self.prior = nn.Parameter(
            torch.randn(num_prior_tokens, embed_dim) * prior_init_std
        )
        # DETR-style slot identity. Owned next to `prior` since the two
        # parameters are conceptually paired (both are [M, d] per-slot
        # contributions to the Q path). Default off so existing
        # checkpoints / configs are byte-identical.
        if slot_embed:
            self.slot_embed = nn.Parameter(
                _make_slot_embed_init(
                    num_prior_tokens, embed_dim,
                    mode=slot_embed_init,
                    std=slot_embed_init_std,
                )
            )
            self.slot_embed_init_mode = slot_embed_init
        else:
            self.slot_embed = None
            self.slot_embed_init_mode = None
        self.updater = PosteriorUpdater(
            embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            gate=gate,
            gate_init=gate_init,
            qk_norm=qk_norm,
            qkv_spectral_norm=qkv_spectral_norm,
            qkv_spectral_learnable=qkv_spectral_learnable,
            qkv_spectral_gamma_init=qkv_spectral_gamma_init,
            normalize_update=normalize_update,
            update_scale_init=update_scale_init,
        )

    def forward(
        self,
        chm_tokens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            chm_tokens: ``[B, P', d]`` CHM evidence, or ``None`` for pure
                prior-only memory (inference without LiDAR, or CHM-dropout
                samples during training).
            batch_size: required only when ``chm_tokens is None``. Used to
                broadcast the prior bank along the batch dimension.

        Returns:
            ``[B, M, d]`` or ``[B, M + P', d]`` depending on
            ``concat_chm_to_memory`` and whether ``chm_tokens`` is supplied.
        """
        if chm_tokens is None:
            return self.prior.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        B = chm_tokens.shape[0]
        prior_b = self.prior.unsqueeze(0).expand(B, -1, -1).contiguous()
        posterior = self.updater(
            prior_b, chm_tokens, slot_embed=self.slot_embed,
        )                                                                # [B, M, d]
        if self.concat_chm_to_memory:
            return torch.cat([posterior, chm_tokens], dim=1)            # [B, M+P', d]
        return posterior                                                # [B, M, d]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Dinov3HeightModel(nn.Module):
    """CHM-prompted height estimation with a stacked Transformer decoder.

    Pipeline (see module docstring for ASCII diagram)::

        image  ─► backbone ─► img_tokens  ─┐
                                           ├─► decoder stack (N layers) ─► refined
        chm    ─► chm_enc  ─► chm_tokens + 2D PE ─► chm_memory ─────────┘
                                           │
                                           └─► reshape + conv head ─► height_map

    Args:
        num_classes:   output channels of the head (usually 1 for height).
        backbone_type: ``"vitl"`` (D=1024), ``"vitb"`` (D=768), or ``"vitg"`` (D=1536).
        patch_size:    backbone patch size; used to compute token grid size.
        n_layers:      number of stacked decoder layers (default 10).
        n_heads:       number of attention heads in every MHA (default 8).
        ffn_ratio:     FFN hidden dim multiplier (default 4).
        dropout:       dropout in attention + FFN (default 0.1).
        chm_dropout:   probability of replacing CHM with zeros during training.
        aux_chm_recon: if True, attach a :class:`CHMReconHead` that reconstructs
            the (corrupted) input CHM map from the CHM encoder's output tokens
            (autoencoder-style aux loss). The reconstruction loss must be
            picked up on the training side via
            ``forward(..., return_aux=True)``; see :class:`HeightEstimationModule`
            for the wiring. Only instantiated when this flag is on, so the default
            ``False`` path is byte-identical to the non-aux model.
        cross_attn_window: radius ``k`` of the locality prior on cross-attention.
            Each image query is allowed to attend only to CHM tokens inside a
            ``(2k+1) × (2k+1)`` window centered at its own grid coordinates.
            Default is ``3`` — a ``7 × 7`` window covers our realistic
            misalignment budget while cutting cross-attn compute by ~20× versus
            the unrestricted case. Set to a value ≥ ``max(h_grid, w_grid)`` for
            an effectively unrestricted window. See Plan v1 for the full
            justification.
        layer_scale_init: initial value for the per-channel LayerScale gain
            applied to every sub-block output before the residual add
            (CaiT / DINOv2 / DINOv3 style). Set to ``None`` (default) to
            **disable LayerScale completely** — no parameters or extra
            multiply are added. Set to a small float such as ``1e-5`` to
            enable it; this is the standard ablation handle.

    Forward:
        ``model(image, chm)`` returns ``height_map`` ``[B, num_classes, H, W]``.
        ``model(image, chm, return_intermediates=True)`` returns
        ``(height_map, intermediates_dict)``.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        patch_size: int = 16,
        load_pretrained_backbone: bool = False,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chm_dropout: float = 0.2,
        aux_chm_recon: bool = False,
        cross_attn_window: int = 3,
        layer_scale_init: Optional[float] = None,
    ):
        super().__init__()
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        if backbone_type not in embedding_dims:
            raise ValueError(f"backbone_type must be one of {list(embedding_dims)}")
        embed_dim = embedding_dims[backbone_type]

        if cross_attn_window < 0:
            raise ValueError(
                f"cross_attn_window must be >= 0, got {cross_attn_window}"
            )

        if layer_scale_init is not None and layer_scale_init <= 0:
            raise ValueError(
                f"layer_scale_init must be None or a positive float, "
                f"got {layer_scale_init}"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.chm_dropout = chm_dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cross_attn_window = cross_attn_window
        self.layer_scale_init = layer_scale_init
        self.aux_chm_recon = aux_chm_recon

        # Lazy cache of window masks keyed by (h, w, k, device.type, device.index).
        # Built on first use at each unique image-grid size + device so we pay
        # the O(P^2) cost once per training run rather than per forward.
        self._window_mask_cache: dict = {}

        # --- Backbone (frozen) ---
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=False,
            return_class_token=False,
            load_pretrained_backbone=load_pretrained_backbone,
        )

        # --- CHM encoder ---
        self.chm_encoder = CHMPromptEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # --- Auxiliary CHM reconstruction head (optional) ---
        self.chm_recon_head = (
            CHMReconHead(in_dim=embed_dim) if aux_chm_recon else None
        )

        # --- Transformer decoder stack ---
        self.decoder = HeightDecoderStack(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
        )

        # --- Conv head ---
        self.head = SimpleConvHead(in_channels=embed_dim, num_classes=num_classes)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        chm: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_aux: bool = False,
    ):
        """
        Args:
            image: ``[B, 3, H, W]`` RGB input.
            chm:   ``[B, 1, H, W]`` old/corrupted CHM (or ``None`` for image-only).
            return_intermediates: if True, also return a dict with all tensor stages.
            return_aux: if True, also return an auxiliary dict with
                ``chm_recon`` (``[B, 1, H, W]`` or ``None``) and
                ``chm_was_dropped`` (``[B]`` bool). Only non-trivial when the
                model was built with ``aux_chm_recon=True``.

        Returns:
            ``height_map`` (``[B, num_classes, H, W]``), or tuples depending
            on the flags:
            - ``(height_map, intermediates)`` if ``return_intermediates=True``.
            - ``(height_map, aux)`` if ``return_aux=True``.
            - ``(height_map, intermediates, aux)`` if both are set.
        """
        B, _, H, W = image.shape
        h_img = H // self.patch_size
        w_img = W // self.patch_size

        # 1. Image tokens from frozen DINOv3
        feats = self.backbone(image)           # list of [B, P, D]; last is deepest
        img_tokens = feats[-1]                 # [B, P, D]

        # 2. CHM tokens + CHM dropout + 2D sin PE
        use_chm = chm is not None
        if use_chm and self.training and torch.rand((), device=chm.device).item() < self.chm_dropout:
            chm = torch.zeros_like(chm)

        # Per-sample "was the CHM dropped out" flag. Covers both the
        # dataset-level full_dropout and the model-level chm_dropout — both
        # collapse CHM to all-zeros before the encoder. Used to mask samples
        # out of the auxiliary reconstruction loss.
        chm_was_dropped = (
            (chm.reshape(B, -1) == 0).all(dim=1)
            if use_chm
            else torch.ones(B, dtype=torch.bool, device=image.device)
        )

        chm_recon: Optional[torch.Tensor] = None
        if use_chm:
            chm_tokens, h_chm, w_chm = self.chm_encoder(chm)  # [B, P', D]

            if return_aux and self.chm_recon_head is not None:
                tokens_grid = (
                    chm_tokens.transpose(1, 2).reshape(B, self.embed_dim, h_chm, w_chm)
                )
                chm_recon = self.chm_recon_head(tokens_grid)
                if chm_recon.shape[-2:] != (H, W):
                    chm_recon = F.interpolate(
                        chm_recon, size=(H, W), mode="bilinear", align_corners=False,
                    )

            # Positional encoding
            pe = build_2d_sincos_pe(
                h_chm, w_chm, self.embed_dim,
                device=chm_tokens.device, dtype=chm_tokens.dtype,
            )                                                 # [P', D]
            chm_memory = chm_tokens + pe.unsqueeze(0)         # broadcast over batch
        else:
            # Degenerate memory: zeros with matching grid size.
            # Cross-attention over a zero memory acts as a no-op after softmax+proj.
            chm_memory = torch.zeros(B, h_img * w_img, self.embed_dim,
                                     device=img_tokens.device, dtype=img_tokens.dtype)

        # 3. Decoder stack with windowed cross-attention.
        #    Mask is [P, P] bool; True entries are blocked (locality prior).
        cross_attn_mask = get_cached_window_mask(
            self._window_mask_cache, h_img, w_img,
            self.cross_attn_window, img_tokens.device,
        )

        if return_intermediates:
            refined, per_layer = self.decoder(
                img_tokens, chm_memory, cross_attn_mask,
                return_intermediates=True,
            )
        else:
            refined = self.decoder(img_tokens, chm_memory, cross_attn_mask)

        # 4. Reshape tokens to spatial + conv head
        D = refined.shape[-1]
        spatial = refined.transpose(1, 2).reshape(B, D, h_img, w_img)  # [B, D, h, w]
        height_map = self.head(spatial, H, W)                          # [B, num_classes, H, W]

        intermediates = None
        if return_intermediates:
            intermediates = {
                "img_tokens": img_tokens,                  # [B, P,  D]
                "chm_memory": chm_memory,                  # [B, P', D]
                "cross_attn_mask": cross_attn_mask,        # [P, P] bool
                "decoder_per_layer": per_layer,            # list[dict] — per-sub-block tensors
                "refined_tokens": refined,                 # [B, P, D] after all N layers
                "decoder_spatial": spatial,                # [B, D, h, w]
            }
        aux = {"chm_recon": chm_recon, "chm_was_dropped": chm_was_dropped} if return_aux else None

        if return_intermediates and return_aux:
            return height_map, intermediates, aux
        if return_intermediates:
            return height_map, intermediates
        if return_aux:
            return height_map, aux
        return height_map

    # ------------------------------------------------------------------
    # Introspection helper
    # ------------------------------------------------------------------
    def architecture_summary(self) -> str:
        """Human-readable architecture tree with shapes and parameter counts."""
        def fmt_n(n: int) -> str:
            for unit in ("", "K", "M", "B"):
                if abs(n) < 1000:
                    return f"{n:.1f}{unit}" if unit else f"{n}"
                n /= 1000
            return f"{n:.1f}T"

        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        D = self.embed_dim
        k = self.cross_attn_window
        win_desc = f"windowed, k={k}  →  (2k+1)×(2k+1) = {(2*k+1)**2} tokens per query"
        if self.layer_scale_init is None:
            ls_desc = "LayerScale: OFF (nn.Identity)"
            ls_wrap_cross = "x = x + MHA(LN(x), LN(chm_memory))"
            ls_wrap_self = "x = x + MHA(LN(x), LN(x))"
            ls_wrap_ffn = "x = x + FFN(LN(x))   [D → 4D → D]"
        else:
            ls_desc = f"LayerScale: ON  (γ init = {self.layer_scale_init:g}, per-channel, learnable)"
            ls_wrap_cross = "x = x + γ_c * MHA(LN(x), LN(chm_memory))"
            ls_wrap_self = "x = x + γ_s * MHA(LN(x), LN(x))"
            ls_wrap_ffn = "x = x + γ_f * FFN(LN(x))   [D → 4D → D]"

        recon_line = (
            f"├── CHMReconHead (aux)                     chm_tokens → input-CHM recon (autoencoder) [B, 1, H, W]   params: {fmt_n(count(self.chm_recon_head))}"
            if self.chm_recon_head is not None else None
        )

        lines = [
            "Dinov3HeightModel",
            f"├── Dinov3Backbone (frozen)                img → img_tokens [B, P, D={D}]   params: {fmt_n(count(self.backbone))}",
            f"├── CHMPromptEncoder                       chm → chm_tokens [B, P', D={D}]  params: {fmt_n(count(self.chm_encoder))}",
        ]
        if recon_line is not None:
            lines.append(recon_line)
        lines += [
            f"├── HeightDecoderStack (× {self.n_layers} layers)       params: {fmt_n(count(self.decoder))}",
            f"│   └── HeightDecoderLayer                  (repeated {self.n_layers}× identically)",
            f"│       ├── CrossAttnSubBlock               {ls_wrap_cross}    heads={self.n_heads}  cross-attn: {win_desc}",
            f"│       ├── SelfAttnSubBlock                {ls_wrap_self}             heads={self.n_heads}  self-attn:  global",
            f"│       └── FFNSubBlock                     {ls_wrap_ffn}",
            f"│       [{ls_desc}]",
            f"│   └── final LayerNorm",
            f"└── SimpleConvHead                         spatial → height_map [B, C, H, W]  params: {fmt_n(count(self.head))}",
            "",
            f"Total params: {fmt_n(count(self))}   (backbone frozen)",
            f"Trainable    : {fmt_n(sum(p.numel() for p in self.parameters() if p.requires_grad))}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DPT variant: refine 4 backbone taps via a shared decoder, then fuse with DPT
# ---------------------------------------------------------------------------


class Dinov3HeightModelDPT(nn.Module):
    """CHM-prompted height estimation with multi-scale DPT fusion.

    Architecture (top-down)::

        image  ─► Dinov3Backbone (frozen)
                    └► 4 intermediate taps at Depth-Anything indices
                       (vitl → [4, 11, 17, 23]; vitb → [2, 5, 8, 11]; …)
                       each: tokens_i [B, P, D]

        chm    ─► CHMPromptEncoder + 2D sin PE ─► chm_memory [B, P', D]

        ┌─ Parallel-batched refinement (shared weights across the 4 taps) ─┐
        │   stack ──► [4B, P, D]                                           │
        │   repeat chm_memory ──► [4B, P', D]                              │
        │   HeightDecoderStack(stacked, chm_memory_rep, window_mask)       │
        │   split ──► [tokens_i'  for i in 0..3]                           │
        └──────────────────────────────────────────────────────────────────┘

        └► DPTHeadLinear([(tokens_i'),]_i=0..3, patch_h, patch_w)
           ─► height_map [B, num_classes, H, W]

    Why a *shared* decoder?
        "Refine tokens using CHM" is the same operation regardless of backbone
        depth. Sharing weights is the correct inductive bias and keeps the
        decoder parameter count identical to the single-scale
        ``Dinov3HeightModel``; only the forward compute grows 4×.

    Args (inherited from ``Dinov3HeightModel``):
        num_classes, backbone_type, dataset, patch_size, load_pretrained_backbone,
        n_layers, n_heads, ffn_ratio, dropout, chm_dropout, aux_chm_recon,
        cross_attn_window, layer_scale_init.

    Args (DPT-specific):
        dpt_features:     feature channel count inside DPT fusion blocks. 256 is
                          Depth-Anything's default.
        dpt_out_channels: per-tap projection widths. Default ``[256, 512, 1024,
                          1024]`` matches the shallow→deep pyramid used by DPT
                          on DINOv2/v3 backbones.
        dpt_use_bn:       BatchNorm inside the refinenet fusion blocks.
        reinit_dpt_head:  if True (default) the final 1×1 classifier is
                          reinitialised with ``std=0.01`` so the initial height
                          prediction is near zero. Prevents the first optimiser
                          steps from pushing L1 / gradient-matching losses into
                          a bad regime.

    Forward:
        ``model(image, chm)`` returns ``[B, num_classes, H, W]``.
        ``model(image, chm, return_intermediates=True)`` returns
        ``(height_map, dict)`` with the CHM memory, window mask, the 4 refined
        token tensors, and per-layer decoder intermediates.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        patch_size: int = 16,
        load_pretrained_backbone: bool = False,
        # decoder stack (identical to Dinov3HeightModel)
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chm_dropout: float = 0.2,
        aux_chm_recon: bool = False,
        cross_attn_window: int = 3,
        layer_scale_init: Optional[float] = None,
        # NEW: per-path LayerScale init for the cross-attn sub-block.
        # ``None`` (default) inherits ``layer_scale_init``; setting an
        # explicit float overrides only the cross-attn path. Use
        # ``cross_attn_layer_scale_init=0.1`` together with
        # ``layer_scale_init=1e-5`` to give the CHM-conditional residual
        # a 10⁴× higher initial gain than self-attn / FFN, preventing
        # the optimiser from gating cross-attn down before the encoder
        # has anything informative to say.
        cross_attn_layer_scale_init: Optional[float] = None,
        # NEW: per-layer cross-attn CHM-prediction probe (option E).
        # When True, attaches a small ``Linear(embed_dim, 1)`` head per
        # decoder layer that predicts the per-image-patch mean CHM from
        # the cross-attn delta (``LayerScale(MHA(...))`` output, before
        # the residual add). Forces every layer's cross-attn to encode
        # CHM-relevant content, sending gradient back through the keys
        # and values to the chm_encoder. The training loop pulls the
        # per-layer predictions from ``aux["chm_pred_per_layer"]`` and
        # combines them with the per-layer-probe weight from config.
        chm_pred_per_layer: bool = False,
        # MHA stability knobs (see MultiHeadAttention). Applied uniformly to
        # decoder cross-attn and decoder self-attn so the QK-Norm / σReparam
        # constraint is enforced at every trainable attention site. Defaults
        # are no-ops, giving a vanilla pre-LN decoder.
        decoder_cross_attn_qk_norm: bool = False,
        decoder_cross_attn_qkv_spectral_norm: bool = False,
        decoder_self_attn_qk_norm: bool = False,
        decoder_self_attn_qkv_spectral_norm: bool = False,
        # Full σ-reparam stack (mirrors V2DPT/V3DPT). When
        # ``sigma_reparam_apply_to_all_linears=True`` the FFN linears in the
        # decoder are also spectrally reparametrised in addition to the QKV
        # projections selected by the per-attention flags above.
        # ``sigma_reparam_learnable=True`` makes σ-reparam's γ scalar a
        # trainable parameter (default: fixed at 1.0). ``sigma_reparam_gamma_init``
        # selects the γ init mode (``"constant"`` keeps γ=1.0; ``"svd"`` sets
        # γ = σ_max(W_init) so the at-init forward magnitude matches an
        # unparametrised Linear). See ``_apply_spectral_reparam``.
        sigma_reparam_learnable: bool = False,
        sigma_reparam_apply_to_all_linears: bool = False,
        sigma_reparam_gamma_init: str = "constant",
        # DPT head
        dpt_features: int = 256,
        dpt_out_channels: Optional[list] = None,
        dpt_use_bn: bool = False,
        reinit_dpt_head: bool = True,
    ):
        super().__init__()
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        if backbone_type not in embedding_dims:
            raise ValueError(f"backbone_type must be one of {list(embedding_dims)}")
        embed_dim = embedding_dims[backbone_type]

        if cross_attn_window < 0:
            raise ValueError(
                f"cross_attn_window must be >= 0, got {cross_attn_window}"
            )

        if layer_scale_init is not None and layer_scale_init <= 0:
            raise ValueError(
                f"layer_scale_init must be None or a positive float, "
                f"got {layer_scale_init}"
            )
        if (
            cross_attn_layer_scale_init is not None
            and cross_attn_layer_scale_init <= 0
        ):
            raise ValueError(
                f"cross_attn_layer_scale_init must be None or a positive "
                f"float, got {cross_attn_layer_scale_init}"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.chm_dropout = chm_dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cross_attn_window = cross_attn_window
        self.layer_scale_init = layer_scale_init
        self.cross_attn_layer_scale_init = cross_attn_layer_scale_init
        self.dpt_features = dpt_features
        self.aux_chm_recon = aux_chm_recon
        self.chm_pred_per_layer_enabled = bool(chm_pred_per_layer)
        self.sigma_reparam_learnable = bool(sigma_reparam_learnable)
        self.sigma_reparam_apply_to_all_linears = bool(
            sigma_reparam_apply_to_all_linears
        )
        self.sigma_reparam_gamma_init = str(sigma_reparam_gamma_init)

        # Lazy cache of window masks keyed by (h, w, k, device.type, device.index).
        self._window_mask_cache: dict = {}

        # --- Backbone (frozen).  The backbone already exposes the 4 Depth-Anything
        # taps at the right layer indices; we consume all of them here. ---
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=False,
            return_class_token=False,
            load_pretrained_backbone=load_pretrained_backbone,
        )
        # Remember the tap indices purely for introspection / summaries.
        self.tap_indices = list(self.backbone.intermediate_layer_idx[backbone_type])

        # --- CHM encoder ---
        self.chm_encoder = CHMPromptEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # --- Auxiliary CHM reconstruction head (optional) ---
        self.chm_recon_head = (
            CHMReconHead(in_dim=embed_dim) if aux_chm_recon else None
        )

        # --- ONE shared decoder stack applied in parallel to all 4 taps ---
        self.decoder = HeightDecoderStack(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            cross_attn_layer_scale_init=cross_attn_layer_scale_init,
            cross_attn_qk_norm=decoder_cross_attn_qk_norm,
            cross_attn_qkv_spectral_norm=decoder_cross_attn_qkv_spectral_norm,
            self_attn_qk_norm=decoder_self_attn_qk_norm,
            self_attn_qkv_spectral_norm=decoder_self_attn_qkv_spectral_norm,
            qkv_spectral_learnable=self.sigma_reparam_learnable,
            qkv_spectral_gamma_init=self.sigma_reparam_gamma_init,
        )

        if self.sigma_reparam_apply_to_all_linears:
            apply_sigma_reparam_to_all_linears(
                self.decoder,
                learnable=self.sigma_reparam_learnable,
                init=1.0,
                gamma_init_mode=self.sigma_reparam_gamma_init,
                n_power_iterations=1,
            )

        # --- Per-layer cross-attn CHM-prediction probes (option E) ---
        # One Linear(D → 1) per decoder layer, applied to the cross-attn
        # delta (post-LayerScale, pre-residual-add) at each layer. The
        # heads are NOT spectrally re-parametrised — they're tiny output
        # readouts whose gradient is the only thing we care about.
        if self.chm_pred_per_layer_enabled:
            self.chm_pred_heads = nn.ModuleList(
                [nn.Linear(embed_dim, 1) for _ in range(n_layers)]
            )
            for h in self.chm_pred_heads:
                nn.init.zeros_(h.bias)
                nn.init.normal_(h.weight, mean=0.0, std=0.02)
        else:
            self.chm_pred_heads = None

        # --- DPT head ---
        if dpt_out_channels is None:
            dpt_out_channels = [256, 512, 1024, 1024]
        if len(dpt_out_channels) != 4:
            raise ValueError(
                f"dpt_out_channels must have length 4, got {len(dpt_out_channels)}"
            )
        self.head = DPTHeadLinear(
            in_channels=embed_dim,
            features=dpt_features,
            num_classes=num_classes,
            use_bn=dpt_use_bn,
            out_channels=dpt_out_channels,
            use_clstoken=False,          # decoder carries no CLS token
            use_auxiliary=False,         # v1: deep supervision off by design
            patch_size=patch_size,
        )

        if reinit_dpt_head:
            self._reinit_dpt_head()

    # ------------------------------------------------------------------
    # Head initialisation
    # ------------------------------------------------------------------
    def _reinit_dpt_head(self) -> None:
        """Init the final 1×1 classifier so the initial prediction is near zero.

        Mirrors ``DepthAnythingV2Plus._initialize_classification_layers``. Keeps
        the first few optimizer steps calm on L1 / GradientMatching losses,
        which otherwise get jolted by a large-magnitude random head.
        """
        final_conv = self.head.scratch.output_conv2  # Conv2d(features, num_classes, 1)
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        chm: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_aux: bool = False,
    ):
        """
        Args:
            image: ``[B, 3, H, W]``.
            chm:   ``[B, 1, H, W]`` or ``None`` for image-only inference.
            return_intermediates: if True, return ``(height_map, dict)``.
            return_aux: if True, also return ``{"chm_recon", "chm_was_dropped"}``
                (only non-trivial when built with ``aux_chm_recon=True``).
                See :class:`Dinov3HeightModel.forward` for the tuple layout.
        """
        B, _, H, W = image.shape
        h_img = H // self.patch_size
        w_img = W // self.patch_size
        P = h_img * w_img
        D = self.embed_dim

        # 1. Four intermediate backbone taps, each [B, P, D], shallow→deep.
        feats = self.backbone(image)
        if len(feats) != 4:
            raise RuntimeError(
                f"Expected 4 backbone taps, got {len(feats)}. "
                f"Dinov3Backbone.intermediate_layer_idx may have changed."
            )

        # 2. CHM tokens + CHM dropout + 2D sin PE. One memory shared across taps.
        use_chm = chm is not None
        if use_chm and self.training and torch.rand((), device=chm.device).item() < self.chm_dropout:
            chm = torch.zeros_like(chm)

        chm_was_dropped = (
            (chm.reshape(B, -1) == 0).all(dim=1)
            if use_chm
            else torch.ones(B, dtype=torch.bool, device=image.device)
        )

        chm_recon: Optional[torch.Tensor] = None
        if use_chm:
            chm_tokens, h_chm, w_chm = self.chm_encoder(chm)         # [B, P', D]

            if return_aux and self.chm_recon_head is not None:
                tokens_grid = (
                    chm_tokens.transpose(1, 2).reshape(B, D, h_chm, w_chm)
                )
                chm_recon = self.chm_recon_head(tokens_grid)
                if chm_recon.shape[-2:] != (H, W):
                    chm_recon = F.interpolate(
                        chm_recon, size=(H, W), mode="bilinear", align_corners=False,
                    )

            pe = build_2d_sincos_pe(
                h_chm, w_chm, D,
                device=chm_tokens.device, dtype=chm_tokens.dtype,
            )                                                        # [P', D]
            chm_memory = chm_tokens + pe.unsqueeze(0)                # [B, P', D]
        else:
            chm_memory = torch.zeros(
                B, P, D, device=feats[0].device, dtype=feats[0].dtype,
            )

        # 3. Parallel-batched decoder pass.
        #    stack taps: [B, P, D] × 4  →  [4B, P, D]   (along batch, shallow-first)
        #    broadcast chm_memory:       →  [4B, P', D] (same memory 4× across taps)
        stacked = torch.cat(feats, dim=0)                            # [4B, P, D]
        chm_memory_rep = chm_memory.repeat(4, 1, 1)                  # [4B, P', D]

        cross_attn_mask = get_cached_window_mask(
            self._window_mask_cache, h_img, w_img,
            self.cross_attn_window, stacked.device,
        )

        # Per-layer CHM-prediction probe (option E) needs the cross-attn
        # delta from every decoder layer, which is only emitted when
        # ``return_intermediates=True``. Force-enable when the heads
        # exist and probes are wanted (controlled by ``return_aux``).
        need_per_layer = (
            self.chm_pred_per_layer_enabled and return_aux and use_chm
        )
        if return_intermediates or need_per_layer:
            refined_stacked, per_layer = self.decoder(
                stacked, chm_memory_rep, cross_attn_mask,
                return_intermediates=True,
            )
        else:
            refined_stacked = self.decoder(
                stacked, chm_memory_rep, cross_attn_mask,
            )                                                        # [4B, P, D]
            per_layer = None

        # Split back to the 4 tap-specific refined token tensors.
        refined_list = list(torch.split(refined_stacked, B, dim=0))   # 4 × [B, P, D]

        # Per-layer CHM-prediction probe: one Linear per layer applied to
        # the cross-attn delta. We average the prediction across the 4 taps
        # so the loss gets one ``[B, P]`` tensor per layer (each tap sees
        # the same CHM, so the per-tap predictions should agree at
        # convergence — averaging acts as a mild regulariser).
        chm_pred_per_layer: Optional[list] = None
        if need_per_layer and per_layer is not None:
            chm_pred_per_layer = []
            for li, head_li in enumerate(self.chm_pred_heads):
                delta_4b = per_layer[li]["cross_attn_delta"]         # [4B, P, D]
                pred_4b = head_li(delta_4b).squeeze(-1)              # [4B, P]
                # Average across the 4 taps (split → mean).
                pred_taps = pred_4b.reshape(4, B, -1)                # [4, B, P]
                chm_pred_per_layer.append(pred_taps.mean(dim=0))     # [B, P]

        # 4. DPT head consumes ``[(tokens_i,), ]``; order must stay shallow→deep.
        out_features = [(t,) for t in refined_list]
        height_map = self.head(out_features, h_img, w_img)           # [B, C, H, W]

        intermediates = None
        if return_intermediates:
            intermediates = {
                "tap_indices": self.tap_indices,                     # e.g. [4, 11, 17, 23]
                "img_tokens_per_tap": feats,                         # list[ [B, P, D] ]
                "chm_memory": chm_memory,                            # [B, P', D]
                "cross_attn_mask": cross_attn_mask,                  # [P, P']
                "refined_tokens_per_tap": refined_list,              # list[ [B, P, D] ]
                "decoder_per_layer": per_layer,                      # list[dict]
            }
        aux = (
            {
                "chm_recon": chm_recon,
                "chm_was_dropped": chm_was_dropped,
                "chm_pred_per_layer": chm_pred_per_layer,
            }
            if return_aux
            else None
        )

        if return_intermediates and return_aux:
            return height_map, intermediates, aux
        if return_intermediates:
            return height_map, intermediates
        if return_aux:
            return height_map, aux
        return height_map

    # ------------------------------------------------------------------
    # Introspection helper
    # ------------------------------------------------------------------
    def architecture_summary(self) -> str:
        """Human-readable architecture tree with shapes and parameter counts."""
        def fmt_n(n: int) -> str:
            for unit in ("", "K", "M", "B"):
                if abs(n) < 1000:
                    return f"{n:.1f}{unit}" if unit else f"{n}"
                n /= 1000
            return f"{n:.1f}T"

        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        D = self.embed_dim
        k = self.cross_attn_window
        win_desc = f"windowed, k={k}  →  (2k+1)×(2k+1) = {(2*k+1)**2} tokens per query"
        if self.layer_scale_init is None:
            ls_desc = "LayerScale: OFF (nn.Identity)"
            ls_wrap_cross = "x = x + MHA(LN(x), LN(chm_memory))"
            ls_wrap_self = "x = x + MHA(LN(x), LN(x))"
            ls_wrap_ffn = "x = x + FFN(LN(x))   [D → 4D → D]"
        else:
            ls_desc = f"LayerScale: ON  (γ init = {self.layer_scale_init:g}, per-channel, learnable)"
            ls_wrap_cross = "x = x + γ_c * MHA(LN(x), LN(chm_memory))"
            ls_wrap_self = "x = x + γ_s * MHA(LN(x), LN(x))"
            ls_wrap_ffn = "x = x + γ_f * FFN(LN(x))   [D → 4D → D]"

        taps = "+".join(str(i) for i in self.tap_indices)
        recon_line = (
            f"├── CHMReconHead (aux)                     chm_tokens → input-CHM recon (autoencoder) [B, 1, H, W]           params: {fmt_n(count(self.chm_recon_head))}"
            if self.chm_recon_head is not None else None
        )
        lines = [
            "Dinov3HeightModelDPT",
            f"├── Dinov3Backbone (frozen)                img → 4 taps at layers [{taps}], each [B, P, D={D}]   params: {fmt_n(count(self.backbone))}",
            f"├── CHMPromptEncoder                       chm → chm_memory [B, P', D={D}]                      params: {fmt_n(count(self.chm_encoder))}",
        ]
        if recon_line is not None:
            lines.append(recon_line)
        lines += [
            f"├── HeightDecoderStack (× {self.n_layers} layers, SHARED across 4 taps via batch-parallelism)    params: {fmt_n(count(self.decoder))}",
            f"│   (input stacked to [4B, P, D]; chm memory repeated to [4B, P', D])",
            f"│   └── HeightDecoderLayer                  (repeated {self.n_layers}× identically)",
            f"│       ├── CrossAttnSubBlock               {ls_wrap_cross}    heads={self.n_heads}  cross-attn: {win_desc}",
            f"│       ├── SelfAttnSubBlock                {ls_wrap_self}             heads={self.n_heads}  self-attn:  global",
            f"│       └── FFNSubBlock                     {ls_wrap_ffn}",
            f"│       [{ls_desc}]",
            f"│   └── final LayerNorm",
            f"└── DPTHeadLinear (features={self.dpt_features}, use_auxiliary=False)   4 taps → height_map [B, C, H, W]   params: {fmt_n(count(self.head))}",
            f"    ├── projects  (1×1: D → out_channels[i])",
            f"    ├── resize_layers (4× ↑, 2× ↑, identity, 2× ↓)  →  shallow→deep pyramid",
            f"    ├── refinenet{{1..4}} (fusion blocks, bilinear upsample)",
            f"    └── output_conv2 (1×1 → num_classes)   [small-std init]",
            "",
            f"Total params: {fmt_n(count(self))}   (backbone frozen)",
            f"Trainable    : {fmt_n(sum(p.numel() for p in self.parameters() if p.requires_grad))}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan v2 top-level models
# ---------------------------------------------------------------------------
#
# These mirror ``Dinov3HeightModel`` / ``Dinov3HeightModelDPT`` but replace
# the CHM → 2D-sincos-PE → memory path with a :class:`LidarPriorLayer` so
# the image decoder can attend over a learnable prior bank when no CHM is
# provided (and over an evidence-updated posterior when one is).
#
# Everything else — backbone, CHM encoder, optional CHM recon head, decoder
# stack, (conv/DPT) head — is reused verbatim from the Plan v1 classes.
# The Plan v1 classes stay unchanged: old checkpoints load into them
# without modification.
#
# Because the prior bank has no spatial layout, the cross-attention is
# intentionally *global* — the windowed locality mask from Plan v1 does not
# apply (it would misalign the query grid with the M prior slots). When
# ``concat_chm_to_memory=True`` the CHM half of the memory also stays
# unmasked for simplicity; if a future ablation wants to reinstate windowed
# attention over the CHM half it can build a composite mask here.


class Dinov3HeightModelV2(nn.Module):
    """Plan v2 single-tap height estimation with a learnable LiDAR prior.

    Pipeline (top-down)::

        image  ►► Dinov3Backbone (frozen) ──► img_tokens [B, P, D]
        chm    ►► CHMPromptEncoder + 2D sin PE ──► chm_tokens [B, P', D]  (optional)

                            LidarPriorLayer
                               │
                    ┌──── prior [M, D]  ← nn.Parameter
                    │         │
                    │         ├─ chm is None ►► memory = prior_b [B, M, D]
                    │         │
                    │         └─ chm present ►► posterior = prior + x-attn(prior, chm)
                    │                            memory = posterior          [B, M, D]
                    │                            (or concat w/ chm_tokens if enabled)
                    ▼
        img_tokens, memory ──► HeightDecoderStack ──► refined [B, P, D]
                               └── cross-attention is GLOBAL (no window mask
                                   — the prior bank is non-spatial by design)
        refined ──► reshape ──► SimpleConvHead ──► height_map [B, num_classes, H, W]

    Training behaviour:
        - ``chm_dropout`` now takes the **prior-only path** instead of
          zeroing the CHM tensor. This gives the prior real gradient signal
          in the no-CHM regime (the whole point of Plan v2).
        - Dataset-level per-sample CHM corruption / dropout is orthogonal
          and still flows through ``CHMPromptEncoder`` normally.

    Args:
        Mirrors :class:`Dinov3HeightModel` with three additions:
            num_prior_tokens:     size of the prior bank ``M`` (default 64).
            concat_chm_to_memory: if True, concat raw CHM tokens after the
                posterior in the memory tensor ("A+B hybrid"). Default False.
            prior_n_heads:        attention heads in the posterior updater.
                Defaults to ``n_heads`` (same width as decoder attention).

        The ``cross_attn_window`` arg from Plan v1 is accepted but ignored —
        the prior bank has no spatial layout to apply locality against.

    Forward:
        Same contract as :class:`Dinov3HeightModel.forward`:
        ``model(image, chm, return_intermediates=False, return_aux=False)``.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        patch_size: int = 16,
        load_pretrained_backbone: bool = False,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chm_dropout: float = 0.5,
        aux_chm_recon: bool = False,
        cross_attn_window: int = 3,  # accepted for YAML compat; ignored in v2
        layer_scale_init: Optional[float] = None,
        # --- Plan v2 specifics ---
        num_prior_tokens: int = 64,
        concat_chm_to_memory: bool = False,
        prior_n_heads: Optional[int] = None,
        prior_init_std: float = 0.02,
        gate: Optional[str] = None,
        gate_init: Optional[float] = None,
        # --- Deprecated kwargs (accepted for backward compatibility) ---
        normalize_update: Optional[bool] = None,
        update_scale_init: Optional[float] = None,
    ):
        super().__init__()
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        if backbone_type not in embedding_dims:
            raise ValueError(f"backbone_type must be one of {list(embedding_dims)}")
        embed_dim = embedding_dims[backbone_type]

        if layer_scale_init is not None and layer_scale_init <= 0:
            raise ValueError(
                f"layer_scale_init must be None or a positive float, "
                f"got {layer_scale_init}"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.chm_dropout = chm_dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.layer_scale_init = layer_scale_init
        self.aux_chm_recon = aux_chm_recon
        self.num_prior_tokens = num_prior_tokens
        self.concat_chm_to_memory = concat_chm_to_memory
        self._cross_attn_window_ignored = cross_attn_window

        # --- Backbone (frozen) ---
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=False,
            return_class_token=False,
            load_pretrained_backbone=load_pretrained_backbone,
        )

        # --- CHM encoder (reused from Plan v1) ---
        self.chm_encoder = CHMPromptEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # --- Auxiliary CHM reconstruction head (optional, reused) ---
        self.chm_recon_head = (
            CHMReconHead(in_dim=embed_dim) if aux_chm_recon else None
        )

        # --- NEW: learnable LiDAR prior + posterior updater ---
        # Both the new (gate, gate_init) API and the deprecated
        # (normalize_update, update_scale_init) API are forwarded untouched;
        # ``PosteriorUpdater`` does the resolution + validation.
        self.lidar_prior = LidarPriorLayer(
            embed_dim=embed_dim,
            num_prior_tokens=num_prior_tokens,
            n_heads=prior_n_heads if prior_n_heads is not None else n_heads,
            dropout=dropout,
            concat_chm_to_memory=concat_chm_to_memory,
            prior_init_std=prior_init_std,
            gate=gate,
            gate_init=gate_init,
            normalize_update=normalize_update,
            update_scale_init=update_scale_init,
        )

        # --- Transformer decoder stack (reused) ---
        self.decoder = HeightDecoderStack(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
        )

        # --- Conv head (reused) ---
        self.head = SimpleConvHead(in_channels=embed_dim, num_classes=num_classes)

    def forward(
        self,
        image: torch.Tensor,
        chm: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_aux: bool = False,
    ):
        B, _, H, W = image.shape
        h_img = H // self.patch_size
        w_img = W // self.patch_size

        # 1. Image tokens from frozen DINOv3
        feats = self.backbone(image)
        img_tokens = feats[-1]                                  # [B, P, D]

        # 2. Model-level CHM dropout:
        #    in Plan v2 this means "take the prior-only path" (chm=None)
        #    rather than zeroing the CHM tensor — gives the prior real
        #    gradient signal in the no-CHM regime.
        if (
            chm is not None
            and self.training
            and torch.rand((), device=chm.device).item() < self.chm_dropout
        ):
            chm = None

        use_chm = chm is not None

        # 3. CHM → tokens + PE (only when CHM is present)
        chm_tokens_with_pe: Optional[torch.Tensor] = None
        chm_recon: Optional[torch.Tensor] = None
        if use_chm:
            chm_tokens, h_chm, w_chm = self.chm_encoder(chm)    # [B, P', D]

            if return_aux and self.chm_recon_head is not None:
                tokens_grid = (
                    chm_tokens.transpose(1, 2)
                    .reshape(B, self.embed_dim, h_chm, w_chm)
                )
                chm_recon = self.chm_recon_head(tokens_grid)
                if chm_recon.shape[-2:] != (H, W):
                    chm_recon = F.interpolate(
                        chm_recon, size=(H, W), mode="bilinear", align_corners=False,
                    )

            pe = build_2d_sincos_pe(
                h_chm, w_chm, self.embed_dim,
                device=chm_tokens.device, dtype=chm_tokens.dtype,
            )
            chm_tokens_with_pe = chm_tokens + pe.unsqueeze(0)   # [B, P', D]

        chm_was_dropped = (
            torch.zeros(B, dtype=torch.bool, device=image.device)
            if use_chm
            else torch.ones(B, dtype=torch.bool, device=image.device)
        )

        # 4. Learnable prior + posterior update. Handles both regimes.
        chm_memory = self.lidar_prior(
            chm_tokens=chm_tokens_with_pe, batch_size=B,
        )                                                       # [B, M, D] or [B, M+P', D]

        # 5. Decoder stack. Cross-attention is global — no window mask
        #    because the prior bank has no spatial layout.
        if return_intermediates:
            refined, per_layer = self.decoder(
                img_tokens, chm_memory, None,
                return_intermediates=True,
            )
        else:
            refined = self.decoder(img_tokens, chm_memory, None)

        # 6. Reshape tokens to spatial + conv head
        D = refined.shape[-1]
        spatial = refined.transpose(1, 2).reshape(B, D, h_img, w_img)
        height_map = self.head(spatial, H, W)

        intermediates = None
        if return_intermediates:
            intermediates = {
                "img_tokens": img_tokens,
                "chm_tokens_with_pe": chm_tokens_with_pe,
                "chm_memory": chm_memory,
                "prior_bank": self.lidar_prior.prior.detach(),
                "decoder_per_layer": per_layer,
                "refined_tokens": refined,
                "decoder_spatial": spatial,
            }
        aux = (
            {"chm_recon": chm_recon, "chm_was_dropped": chm_was_dropped}
            if return_aux
            else None
        )

        if return_intermediates and return_aux:
            return height_map, intermediates, aux
        if return_intermediates:
            return height_map, intermediates
        if return_aux:
            return height_map, aux
        return height_map

    def architecture_summary(self) -> str:
        """Human-readable architecture tree with shapes and parameter counts."""
        def fmt_n(n: int) -> str:
            for unit in ("", "K", "M", "B"):
                if abs(n) < 1000:
                    return f"{n:.1f}{unit}" if unit else f"{n}"
                n /= 1000
            return f"{n:.1f}T"

        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        D = self.embed_dim
        M = self.num_prior_tokens
        concat_desc = (
            "posterior + raw CHM concat (A+B hybrid)"
            if self.concat_chm_to_memory
            else "pure posterior (Option B)"
        )
        ls_desc = (
            "LayerScale: OFF (nn.Identity)"
            if self.layer_scale_init is None
            else f"LayerScale: ON  (γ init = {self.layer_scale_init:g})"
        )

        recon_line = (
            f"├── CHMReconHead (aux)                     chm_tokens → input-CHM recon (autoencoder)   params: {fmt_n(count(self.chm_recon_head))}"
            if self.chm_recon_head is not None else None
        )

        lines = [
            "Dinov3HeightModelV2  (Plan v2 — learnable LiDAR prior)",
            f"├── Dinov3Backbone (frozen)                img → img_tokens [B, P, D={D}]   params: {fmt_n(count(self.backbone))}",
            f"├── CHMPromptEncoder                       chm → chm_tokens [B, P', D={D}]  params: {fmt_n(count(self.chm_encoder))}",
        ]
        if recon_line is not None:
            lines.append(recon_line)
        lines += [
            f"├── LidarPriorLayer                        memory = {concat_desc}   params: {fmt_n(count(self.lidar_prior))}",
            f"│   ├── prior    nn.Parameter [M={M}, D={D}]",
            f"│   └── PosteriorUpdater  (prior queries CHM, residual update)",
            f"├── HeightDecoderStack (× {self.n_layers} layers, GLOBAL cross-attn)  params: {fmt_n(count(self.decoder))}",
            f"│   [{ls_desc}]",
            f"└── SimpleConvHead                         spatial → height_map [B, C, H, W]  params: {fmt_n(count(self.head))}",
            "",
            f"Total params: {fmt_n(count(self))}   (backbone frozen)",
            f"Trainable    : {fmt_n(sum(p.numel() for p in self.parameters() if p.requires_grad))}",
        ]
        return "\n".join(lines)


class Dinov3HeightModelV2DPT(nn.Module):
    """Plan v2 multi-scale (DPT) height estimation with a learnable LiDAR prior.

    Architecture combines:
        - Plan v2 memory from :class:`LidarPriorLayer` (prior bank +
          posterior update).
        - 4-tap Depth-Anything-style multi-scale fusion via
          :class:`model.depth_dpt.DPTHeadLinear`.

    Relative to :class:`Dinov3HeightModelDPT`, only the ``chm_memory``
    construction is swapped for the LidarPriorLayer path; the 4-tap
    backbone forward, shared-decoder batch-parallelism, and DPT head are
    reused unchanged.

    Args:
        Mirrors :class:`Dinov3HeightModelDPT` with the same Plan v2
        additions as :class:`Dinov3HeightModelV2`:
            num_prior_tokens, concat_chm_to_memory, prior_n_heads.
        ``cross_attn_window`` is accepted for YAML-config compatibility but
        ignored (no spatial locality on the prior bank).
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        patch_size: int = 16,
        load_pretrained_backbone: bool = False,
        # decoder stack
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chm_dropout: float = 0.5,
        aux_chm_recon: bool = False,
        cross_attn_window: int = 3,  # accepted for YAML compat; ignored in v2
        layer_scale_init: Optional[float] = None,
        # NEW: per-path LayerScale init for the cross-attn sub-block.
        # ``None`` (default) inherits ``layer_scale_init``; setting an
        # explicit float (e.g. 0.1) overrides only the cross-attn path so
        # the CHM-conditional residual starts with a higher gain than
        # self-attn / FFN. Designed to address the W6 finding that
        # ``γ_cross`` was 3-5× smaller than ``γ_self`` and ``γ_ffn``
        # because the optimiser actively suppresses cross-attn when the
        # CHM keys are rank-collapsed.
        cross_attn_layer_scale_init: Optional[float] = None,
        # NEW: per-layer cross-attn CHM-prediction probe (option E).
        # See ``Dinov3HeightModelDPT`` for the full description; same
        # semantics here. Forces every decoder layer's cross-attn output
        # to carry CHM-relevant content so gradient flows back through
        # the keys/values to ``chm_encoder``.
        chm_pred_per_layer: bool = False,
        # DPT head
        dpt_features: int = 256,
        dpt_out_channels: Optional[list] = None,
        dpt_use_bn: bool = False,
        reinit_dpt_head: bool = True,
        # --- Plan v2 specifics ---
        num_prior_tokens: int = 64,
        concat_chm_to_memory: bool = False,
        prior_n_heads: Optional[int] = None,
        prior_init_std: float = 0.02,
        # NEW — DETR-style learnable slot identity. See
        # :class:`LidarPriorLayer` for the full rationale and the available
        # ``slot_embed_init`` modes.
        slot_embed: bool = False,
        slot_embed_init: str = "random",
        slot_embed_init_std: float = 0.02,
        gate: Optional[str] = None,
        gate_init: Optional[float] = None,
        # MHA stability knobs (see MultiHeadAttention). Applied uniformly
        # across decoder cross-attn, decoder self-attn, and the prior /
        # posterior updater so the QK-Norm / σReparam constraint is enforced
        # at every trainable attention site. Defaults are no-ops.
        decoder_cross_attn_qk_norm: bool = False,
        decoder_cross_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        decoder_self_attn_qk_norm: bool = False,
        decoder_self_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        updater_qk_norm: bool = False,
        updater_qkv_spectral_norm: Union[bool, float, None] = False,
        # See ``Dinov3HeightModelV3DPT`` for full documentation; same
        # semantics apply here. ``decoder_inter_call_ln`` is V3-only and
        # is therefore omitted from V2.
        sigma_reparam_learnable: bool = False,
        sigma_reparam_apply_to_all_linears: bool = False,
        # NEW (U-series): γ initialisation mode for σReparam. ``"constant"``
        # (paper's NLP recipe, Table 13) keeps γ=1.0 at init; ``"svd"``
        # (paper's vision recipe, §4.2; U-series default) initialises γ to
        # σ_max(W_init) per linear so the at-init forward magnitude matches
        # an unparametrised Linear. Only meaningful when
        # ``sigma_reparam_learnable=True``. See _apply_spectral_reparam.
        sigma_reparam_gamma_init: str = "constant",
        # --- Deprecated kwargs (accepted for backward compatibility) ---
        normalize_update: Optional[bool] = None,
        update_scale_init: Optional[float] = None,
    ):
        super().__init__()
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        if backbone_type not in embedding_dims:
            raise ValueError(f"backbone_type must be one of {list(embedding_dims)}")
        embed_dim = embedding_dims[backbone_type]

        if layer_scale_init is not None and layer_scale_init <= 0:
            raise ValueError(
                f"layer_scale_init must be None or a positive float, "
                f"got {layer_scale_init}"
            )
        if (
            cross_attn_layer_scale_init is not None
            and cross_attn_layer_scale_init <= 0
        ):
            raise ValueError(
                f"cross_attn_layer_scale_init must be None or a positive "
                f"float, got {cross_attn_layer_scale_init}"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.chm_dropout = chm_dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.layer_scale_init = layer_scale_init
        self.cross_attn_layer_scale_init = cross_attn_layer_scale_init
        self.dpt_features = dpt_features
        self.aux_chm_recon = aux_chm_recon
        self.chm_pred_per_layer_enabled = bool(chm_pred_per_layer)
        self.num_prior_tokens = num_prior_tokens
        self.concat_chm_to_memory = concat_chm_to_memory
        self._cross_attn_window_ignored = cross_attn_window
        self.sigma_reparam_learnable = bool(sigma_reparam_learnable)
        self.sigma_reparam_apply_to_all_linears = bool(
            sigma_reparam_apply_to_all_linears
        )
        self.sigma_reparam_gamma_init = str(sigma_reparam_gamma_init)

        # --- Backbone (frozen) ---
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=False,
            return_class_token=False,
            load_pretrained_backbone=load_pretrained_backbone,
        )
        self.tap_indices = list(self.backbone.intermediate_layer_idx[backbone_type])

        # --- CHM encoder + optional recon head (reused) ---
        self.chm_encoder = CHMPromptEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.chm_recon_head = (
            CHMReconHead(in_dim=embed_dim) if aux_chm_recon else None
        )

        # --- NEW: learnable LiDAR prior + posterior updater ---
        self.lidar_prior = LidarPriorLayer(
            embed_dim=embed_dim,
            num_prior_tokens=num_prior_tokens,
            n_heads=prior_n_heads if prior_n_heads is not None else n_heads,
            dropout=dropout,
            concat_chm_to_memory=concat_chm_to_memory,
            prior_init_std=prior_init_std,
            slot_embed=slot_embed,
            slot_embed_init=slot_embed_init,
            slot_embed_init_std=slot_embed_init_std,
            gate=gate,
            gate_init=gate_init,
            qk_norm=updater_qk_norm,
            qkv_spectral_norm=updater_qkv_spectral_norm,
            qkv_spectral_learnable=self.sigma_reparam_learnable,
            qkv_spectral_gamma_init=self.sigma_reparam_gamma_init,
            normalize_update=normalize_update,
            update_scale_init=update_scale_init,
        )

        # --- Shared decoder across 4 taps (reused) ---
        self.decoder = HeightDecoderStack(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            cross_attn_layer_scale_init=cross_attn_layer_scale_init,
            cross_attn_qk_norm=decoder_cross_attn_qk_norm,
            cross_attn_qkv_spectral_norm=decoder_cross_attn_qkv_spectral_norm,
            self_attn_qk_norm=decoder_self_attn_qk_norm,
            self_attn_qkv_spectral_norm=decoder_self_attn_qkv_spectral_norm,
            qkv_spectral_learnable=self.sigma_reparam_learnable,
            qkv_spectral_gamma_init=self.sigma_reparam_gamma_init,
        )

        # Extend σReparam coverage to all remaining linears in
        # decoder + lidar_prior (skips already-parametrised QKV).
        if self.sigma_reparam_apply_to_all_linears:
            n_dec = apply_sigma_reparam_to_all_linears(
                self.decoder,
                learnable=self.sigma_reparam_learnable,
                init=1.0,
                gamma_init_mode=self.sigma_reparam_gamma_init,
                n_power_iterations=1,
            )
            n_pri = apply_sigma_reparam_to_all_linears(
                self.lidar_prior,
                learnable=self.sigma_reparam_learnable,
                init=1.0,
                gamma_init_mode=self.sigma_reparam_gamma_init,
                n_power_iterations=1,
            )
            self._sigma_reparam_extended_count = int(n_dec + n_pri)
        else:
            self._sigma_reparam_extended_count = 0

        # --- Per-layer cross-attn CHM-prediction probes (option E) ---
        if self.chm_pred_per_layer_enabled:
            self.chm_pred_heads = nn.ModuleList(
                [nn.Linear(embed_dim, 1) for _ in range(n_layers)]
            )
            for h in self.chm_pred_heads:
                nn.init.zeros_(h.bias)
                nn.init.normal_(h.weight, mean=0.0, std=0.02)
        else:
            self.chm_pred_heads = None

        # --- DPT head (reused) ---
        if dpt_out_channels is None:
            dpt_out_channels = [256, 512, 1024, 1024]
        if len(dpt_out_channels) != 4:
            raise ValueError(
                f"dpt_out_channels must have length 4, got {len(dpt_out_channels)}"
            )
        self.head = DPTHeadLinear(
            in_channels=embed_dim,
            features=dpt_features,
            num_classes=num_classes,
            use_bn=dpt_use_bn,
            out_channels=dpt_out_channels,
            use_clstoken=False,
            use_auxiliary=False,
            patch_size=patch_size,
        )
        if reinit_dpt_head:
            self._reinit_dpt_head()

    def _reinit_dpt_head(self) -> None:
        """Small-std init on the final 1×1 classifier (same as Plan v1 DPT)."""
        final_conv = self.head.scratch.output_conv2
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(
        self,
        image: torch.Tensor,
        chm: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_aux: bool = False,
    ):
        B, _, H, W = image.shape
        h_img = H // self.patch_size
        w_img = W // self.patch_size
        D = self.embed_dim

        # 1. Four intermediate backbone taps
        feats = self.backbone(image)
        if len(feats) != 4:
            raise RuntimeError(
                f"Expected 4 backbone taps, got {len(feats)}. "
                f"Dinov3Backbone.intermediate_layer_idx may have changed."
            )

        # 2. Model-level CHM dropout → prior-only path (Plan v2 semantics).
        if (
            chm is not None
            and self.training
            and torch.rand((), device=chm.device).item() < self.chm_dropout
        ):
            chm = None
        use_chm = chm is not None

        # 3. CHM → tokens + PE (only when CHM is present)
        chm_tokens_with_pe: Optional[torch.Tensor] = None
        chm_recon: Optional[torch.Tensor] = None
        if use_chm:
            chm_tokens, h_chm, w_chm = self.chm_encoder(chm)        # [B, P', D]

            if return_aux and self.chm_recon_head is not None:
                tokens_grid = (
                    chm_tokens.transpose(1, 2).reshape(B, D, h_chm, w_chm)
                )
                chm_recon = self.chm_recon_head(tokens_grid)
                if chm_recon.shape[-2:] != (H, W):
                    chm_recon = F.interpolate(
                        chm_recon, size=(H, W), mode="bilinear", align_corners=False,
                    )

            pe = build_2d_sincos_pe(
                h_chm, w_chm, D,
                device=chm_tokens.device, dtype=chm_tokens.dtype,
            )
            chm_tokens_with_pe = chm_tokens + pe.unsqueeze(0)       # [B, P', D]

        chm_was_dropped = (
            torch.zeros(B, dtype=torch.bool, device=image.device)
            if use_chm
            else torch.ones(B, dtype=torch.bool, device=image.device)
        )

        # 4. Learnable prior + posterior update. Same memory used across the
        #    4 taps (just as Plan v1 DPT shares chm_memory across taps).
        chm_memory = self.lidar_prior(
            chm_tokens=chm_tokens_with_pe, batch_size=B,
        )                                                            # [B, M, D] or [B, M+P', D]

        # 5. Parallel-batched decoder pass across the 4 taps.
        stacked = torch.cat(feats, dim=0)                            # [4B, P, D]
        chm_memory_rep = chm_memory.repeat(4, 1, 1)                  # [4B, M(+P'), D]

        # Force-enable layer intermediates when the per-layer probe needs
        # them; otherwise honour the caller's ``return_intermediates``.
        need_per_layer = (
            self.chm_pred_per_layer_enabled and return_aux and use_chm
        )
        # Global cross-attention — no window mask on a non-spatial prior bank.
        if return_intermediates or need_per_layer:
            refined_stacked, per_layer = self.decoder(
                stacked, chm_memory_rep, None,
                return_intermediates=True,
            )
        else:
            refined_stacked = self.decoder(
                stacked, chm_memory_rep, None,
            )                                                        # [4B, P, D]
            per_layer = None

        refined_list = list(torch.split(refined_stacked, B, dim=0))  # 4 × [B, P, D]

        # Per-layer CHM-prediction probe (option E).
        chm_pred_per_layer: Optional[list] = None
        if need_per_layer and per_layer is not None:
            chm_pred_per_layer = []
            for li, head_li in enumerate(self.chm_pred_heads):
                delta_4b = per_layer[li]["cross_attn_delta"]         # [4B, P, D]
                pred_4b = head_li(delta_4b).squeeze(-1)              # [4B, P]
                pred_taps = pred_4b.reshape(4, B, -1)                # [4, B, P]
                chm_pred_per_layer.append(pred_taps.mean(dim=0))     # [B, P]

        out_features = [(t,) for t in refined_list]
        height_map = self.head(out_features, h_img, w_img)           # [B, C, H, W]

        intermediates = None
        if return_intermediates:
            intermediates = {
                "tap_indices": self.tap_indices,
                "img_tokens_per_tap": feats,
                "chm_tokens_with_pe": chm_tokens_with_pe,
                "chm_memory": chm_memory,
                "prior_bank": self.lidar_prior.prior.detach(),
                "refined_tokens_per_tap": refined_list,
                "decoder_per_layer": per_layer,
            }
        aux = (
            {
                "chm_recon": chm_recon,
                "chm_was_dropped": chm_was_dropped,
                "chm_pred_per_layer": chm_pred_per_layer,
            }
            if return_aux
            else None
        )

        if return_intermediates and return_aux:
            return height_map, intermediates, aux
        if return_intermediates:
            return height_map, intermediates
        if return_aux:
            return height_map, aux
        return height_map

    def architecture_summary(self) -> str:
        """Human-readable architecture tree with shapes and parameter counts."""
        def fmt_n(n: int) -> str:
            for unit in ("", "K", "M", "B"):
                if abs(n) < 1000:
                    return f"{n:.1f}{unit}" if unit else f"{n}"
                n /= 1000
            return f"{n:.1f}T"

        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        D = self.embed_dim
        M = self.num_prior_tokens
        taps = "+".join(str(i) for i in self.tap_indices)
        concat_desc = (
            "posterior + raw CHM concat (A+B hybrid)"
            if self.concat_chm_to_memory
            else "pure posterior (Option B)"
        )
        ls_desc = (
            "LayerScale: OFF (nn.Identity)"
            if self.layer_scale_init is None
            else f"LayerScale: ON  (γ init = {self.layer_scale_init:g})"
        )

        recon_line = (
            f"├── CHMReconHead (aux)                     chm_tokens → input-CHM recon (autoencoder)   params: {fmt_n(count(self.chm_recon_head))}"
            if self.chm_recon_head is not None else None
        )
        lines = [
            "Dinov3HeightModelV2DPT  (Plan v2 + DPT multi-scale fusion)",
            f"├── Dinov3Backbone (frozen)                img → 4 taps at layers [{taps}]   params: {fmt_n(count(self.backbone))}",
            f"├── CHMPromptEncoder                       chm → chm_tokens [B, P', D={D}]   params: {fmt_n(count(self.chm_encoder))}",
        ]
        if recon_line is not None:
            lines.append(recon_line)
        lines += [
            f"├── LidarPriorLayer                        memory = {concat_desc}   params: {fmt_n(count(self.lidar_prior))}",
            f"│   ├── prior    nn.Parameter [M={M}, D={D}]",
            f"│   └── PosteriorUpdater  (prior queries CHM, residual update)",
            f"├── HeightDecoderStack (× {self.n_layers} layers, SHARED across 4 taps)   params: {fmt_n(count(self.decoder))}",
            f"│   (stacked to [4B, P, D]; chm memory repeated to [4B, M(+P'), D])",
            f"│   [{ls_desc}]",
            f"└── DPTHeadLinear (features={self.dpt_features})   4 taps → height_map   params: {fmt_n(count(self.head))}",
            "",
            f"Total params: {fmt_n(count(self))}   (backbone frozen)",
            f"Trainable    : {fmt_n(sum(p.numel() for p in self.parameters() if p.requires_grad))}",
        ]
        return "\n".join(lines)


# =====================================================================
# Plan v3 — per-layer prior refinement (DETR-style shared updater)
# =====================================================================
#
# V2 runs the PosteriorUpdater once, then feeds the same posterior to
# all 10 decoder layers. V3 runs the updater at *every* decoder layer
# using shared weights (single PosteriorUpdater instance, called N
# times). The prior evolves layer-by-layer against fixed CHM evidence.
#
# This gives the updater 10x the effective depth without 10x the
# parameters — the same trick DETR uses for its decoder.


class HeightDecoderStackV3(nn.Module):
    """V3 decoder: per-layer prior refinement with a shared updater.

    Combines the roles of ``LidarPriorLayer`` and ``HeightDecoderStack``
    into one module. Owns the prior bank, the shared updater, and the
    decoder layers.

    Per-layer loop::

        posterior_0 = prior.expand(B)
        for layer_i in layers:
            if chm_tokens is not None:
                posterior_i = updater(posterior_{i-1}, chm_tokens)
            x = layer_i(x, posterior_i)
        return final_norm(x), posterior_N
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        layer_scale_init: Optional[float] = None,
        cross_attn_layer_scale_init: Optional[float] = None,
        num_prior_tokens: int = 64,
        prior_init_std: float = 0.02,
        slot_embed: bool = False,
        slot_embed_init: str = "random",
        slot_embed_init_std: float = 0.02,
        prior_n_heads: Optional[int] = None,
        gate: Optional[str] = None,
        gate_init: Optional[float] = None,
        # MHA stability knobs — applied uniformly across decoder cross-attn,
        # decoder self-attn, and the shared updater's cross-attn so QK-Norm
        # / σReparam constraints are enforced at every trainable attention
        # site. Defaults are no-ops.
        cross_attn_qk_norm: bool = False,
        cross_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        self_attn_qk_norm: bool = False,
        self_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        updater_qk_norm: bool = False,
        updater_qkv_spectral_norm: Union[bool, float, None] = False,
        qkv_spectral_learnable: bool = False,
        qkv_spectral_gamma_init: str = "constant",
        # When True, insert a fresh LayerNorm after each updater call
        # (V3 only). The renormalised posterior is used both as the
        # decoder-layer's cross-attention memory and as the input to the
        # *next* updater iteration. Provides a structural cap on
        # cumulative magnitude blowup from the shared updater's gate
        # compounding (see docs/chm/insights/v3_shared_updater_compounding.md).
        # ``Identity`` when False — no extra params, no behaviour change.
        inter_call_ln: bool = False,
        # --- Deprecated kwargs (accepted for backward compatibility) ---
        normalize_update: Optional[bool] = None,
        update_scale_init: Optional[float] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.num_prior_tokens = num_prior_tokens
        self.embed_dim = embed_dim
        self.inter_call_ln_enabled = bool(inter_call_ln)

        self.prior = nn.Parameter(
            torch.randn(num_prior_tokens, embed_dim) * prior_init_std
        )
        # DETR-style slot identity, reused across all 10 shared updater
        # iterations. Single parameter, no per-iter copies. See
        # :func:`_make_slot_embed_init` for the available init modes
        # (``random`` vs ``sincos_2d``).
        if slot_embed:
            self.slot_embed = nn.Parameter(
                _make_slot_embed_init(
                    num_prior_tokens, embed_dim,
                    mode=slot_embed_init,
                    std=slot_embed_init_std,
                )
            )
            self.slot_embed_init_mode = slot_embed_init
        else:
            self.slot_embed = None
            self.slot_embed_init_mode = None
        self.updater = PosteriorUpdater(
            embed_dim,
            n_heads=prior_n_heads if prior_n_heads is not None else n_heads,
            dropout=dropout,
            gate=gate,
            gate_init=gate_init,
            qk_norm=updater_qk_norm,
            qkv_spectral_norm=updater_qkv_spectral_norm,
            qkv_spectral_learnable=qkv_spectral_learnable,
            qkv_spectral_gamma_init=qkv_spectral_gamma_init,
            normalize_update=normalize_update,
            update_scale_init=update_scale_init,
        )
        self.inter_call_ln = (
            nn.LayerNorm(embed_dim) if self.inter_call_ln_enabled else nn.Identity()
        )
        self.layers = nn.ModuleList([
            HeightDecoderLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                layer_scale_init=layer_scale_init,
                cross_attn_layer_scale_init=cross_attn_layer_scale_init,
                cross_attn_qk_norm=cross_attn_qk_norm,
                cross_attn_qkv_spectral_norm=cross_attn_qkv_spectral_norm,
                self_attn_qk_norm=self_attn_qk_norm,
                self_attn_qkv_spectral_norm=self_attn_qkv_spectral_norm,
                qkv_spectral_learnable=qkv_spectral_learnable,
                qkv_spectral_gamma_init=qkv_spectral_gamma_init,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        chm_tokens: Optional[torch.Tensor],
        cross_attn_mask: Optional[torch.Tensor],
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list, torch.Tensor]]:
        B = x.shape[0]
        posterior = self.prior.unsqueeze(0).expand(B, -1, -1).contiguous()

        per_layer = []
        for layer in self.layers:
            if chm_tokens is not None:
                posterior = self.updater(
                    posterior, chm_tokens, slot_embed=self.slot_embed,
                )
                # Renormalise the cumulative posterior between updater
                # iterations. ``inter_call_ln`` is ``Identity`` when the
                # feature is disabled, preserving the V3 baseline behaviour.
                posterior = self.inter_call_ln(posterior)

            if return_intermediates:
                x, inter = layer(
                    x, posterior, cross_attn_mask,
                    return_intermediates=True,
                )
                per_layer.append(inter)
            else:
                x = layer(x, posterior, cross_attn_mask)

        x = self.final_norm(x)

        if return_intermediates:
            return x, per_layer, posterior
        return x


class Dinov3HeightModelV3DPT(nn.Module):
    """Plan v3 multi-scale height estimation with per-layer prior refinement.

    Relative to V2: the single pre-decoder updater call is replaced by
    per-layer refinement with shared weights (DETR-style). The prior
    evolves through all decoder layers against fixed CHM evidence.
    ``concat_chm_to_memory`` is removed — the decoder always sees the
    posterior, never raw CHM tokens.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        patch_size: int = 16,
        load_pretrained_backbone: bool = False,
        n_layers: int = 10,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        chm_dropout: float = 0.5,
        aux_chm_recon: bool = False,
        cross_attn_window: int = 3,
        layer_scale_init: Optional[float] = None,
        dpt_features: int = 256,
        dpt_out_channels: Optional[list] = None,
        dpt_use_bn: bool = False,
        reinit_dpt_head: bool = True,
        num_prior_tokens: int = 64,
        prior_n_heads: Optional[int] = None,
        prior_init_std: float = 0.02,
        # NEW — DETR-style learnable slot identity, shared across all
        # 10 updater iterations. See :class:`LidarPriorLayer` for the
        # full rationale and the available ``slot_embed_init`` modes.
        slot_embed: bool = False,
        slot_embed_init: str = "random",
        slot_embed_init_std: float = 0.02,
        gate: Optional[str] = None,
        gate_init: Optional[float] = None,
        # MHA stability knobs (see MultiHeadAttention). Applied uniformly
        # across decoder cross-attn, decoder self-attn, and the in-stack
        # posterior updater so the QK-Norm / σReparam constraint is enforced
        # at every trainable attention site. Defaults are no-ops.
        decoder_cross_attn_qk_norm: bool = False,
        decoder_cross_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        decoder_self_attn_qk_norm: bool = False,
        decoder_self_attn_qkv_spectral_norm: Union[bool, float, None] = False,
        updater_qk_norm: bool = False,
        updater_qkv_spectral_norm: Union[bool, float, None] = False,
        # NEW — σReparam mode selector. When True, every per-site
        # ``*_qkv_spectral_norm`` flag installs the *learnable* γ variant
        # from Zhai et al. 2023 (σReparam, eq. 2) instead of the static
        # post-scale (default). γ is a per-projection scalar parameter,
        # initialised to 1.0 (or to the float passed via the per-site
        # flag, e.g. ``decoder_cross_attn_qkv_spectral_norm: 1.0``).
        # No-op when no per-site flag is enabled.
        sigma_reparam_learnable: bool = False,
        # NEW — σReparam coverage. When True, σReparam is additionally
        # registered on *every other* ``nn.Linear`` inside the decoder
        # stack and the shared updater (out_proj, FFN.fc1, FFN.fc2,
        # gate's GRU linears, etc.). Per-site QKV caps are installed
        # first and skipped here so they are never double-wrapped. This
        # matches the paper's "all linear layers" recipe (§4.1) which
        # gave the 12-pt ImageNet improvement; capping QKV alone is a
        # subset of that recipe. ``learnable`` is governed by
        # ``sigma_reparam_learnable``; init is fixed to 1.0 for the
        # extended-coverage layers.
        sigma_reparam_apply_to_all_linears: bool = False,
        # NEW (U-series): γ initialisation mode for σReparam. ``"constant"``
        # (paper's NLP recipe, Table 13) keeps γ=1.0 at init; ``"svd"``
        # (paper's vision recipe, §4.2; U-series default) initialises γ to
        # σ_max(W_init) per linear so the at-init forward magnitude matches
        # an unparametrised Linear. Only meaningful when
        # ``sigma_reparam_learnable=True``. See _apply_spectral_reparam.
        sigma_reparam_gamma_init: str = "constant",
        # NEW — inter-block LayerNorm in the V3 stack. See
        # ``HeightDecoderStackV3.__init__`` for semantics.
        decoder_inter_call_ln: bool = False,
        # --- Deprecated kwargs (accepted for backward compatibility) ---
        normalize_update: Optional[bool] = None,
        update_scale_init: Optional[float] = None,
    ):
        super().__init__()
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        if backbone_type not in embedding_dims:
            raise ValueError(f"backbone_type must be one of {list(embedding_dims)}")
        embed_dim = embedding_dims[backbone_type]

        if layer_scale_init is not None and layer_scale_init <= 0:
            raise ValueError(
                f"layer_scale_init must be None or a positive float, "
                f"got {layer_scale_init}"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.chm_dropout = chm_dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.layer_scale_init = layer_scale_init
        self.dpt_features = dpt_features
        self.aux_chm_recon = aux_chm_recon
        self.num_prior_tokens = num_prior_tokens
        self.concat_chm_to_memory = False
        self._cross_attn_window_ignored = cross_attn_window
        self.sigma_reparam_learnable = bool(sigma_reparam_learnable)
        self.sigma_reparam_apply_to_all_linears = bool(
            sigma_reparam_apply_to_all_linears
        )
        self.sigma_reparam_gamma_init = str(sigma_reparam_gamma_init)
        self.decoder_inter_call_ln = bool(decoder_inter_call_ln)

        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=False,
            return_class_token=False,
            load_pretrained_backbone=load_pretrained_backbone,
        )
        self.tap_indices = list(self.backbone.intermediate_layer_idx[backbone_type])

        self.chm_encoder = CHMPromptEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.chm_recon_head = (
            CHMReconHead(in_dim=embed_dim) if aux_chm_recon else None
        )

        self.decoder = HeightDecoderStackV3(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            num_prior_tokens=num_prior_tokens,
            prior_init_std=prior_init_std,
            slot_embed=slot_embed,
            slot_embed_init=slot_embed_init,
            slot_embed_init_std=slot_embed_init_std,
            prior_n_heads=prior_n_heads,
            gate=gate,
            gate_init=gate_init,
            cross_attn_qk_norm=decoder_cross_attn_qk_norm,
            cross_attn_qkv_spectral_norm=decoder_cross_attn_qkv_spectral_norm,
            self_attn_qk_norm=decoder_self_attn_qk_norm,
            self_attn_qkv_spectral_norm=decoder_self_attn_qkv_spectral_norm,
            updater_qk_norm=updater_qk_norm,
            updater_qkv_spectral_norm=updater_qkv_spectral_norm,
            qkv_spectral_learnable=self.sigma_reparam_learnable,
            qkv_spectral_gamma_init=self.sigma_reparam_gamma_init,
            inter_call_ln=self.decoder_inter_call_ln,
            normalize_update=normalize_update,
            update_scale_init=update_scale_init,
        )

        # Extend σReparam coverage to every remaining ``nn.Linear`` in the
        # decoder stack (out_proj on each MHA, FFN.fc1/fc2, GRU linears,
        # etc.). Layers that already carry a parametrisation — i.e. the
        # fused QKV projections that the per-site path installed above —
        # are skipped automatically, so this never double-wraps.
        if self.sigma_reparam_apply_to_all_linears:
            n_extended = apply_sigma_reparam_to_all_linears(
                self.decoder,
                learnable=self.sigma_reparam_learnable,
                init=1.0,
                gamma_init_mode=self.sigma_reparam_gamma_init,
                n_power_iterations=1,
                skip_already_parametrised=True,
            )
            self._sigma_reparam_extended_count = int(n_extended)
        else:
            self._sigma_reparam_extended_count = 0

        if dpt_out_channels is None:
            dpt_out_channels = [256, 512, 1024, 1024]
        if len(dpt_out_channels) != 4:
            raise ValueError(
                f"dpt_out_channels must have length 4, got {len(dpt_out_channels)}"
            )
        self.head = DPTHeadLinear(
            in_channels=embed_dim,
            features=dpt_features,
            num_classes=num_classes,
            use_bn=dpt_use_bn,
            out_channels=dpt_out_channels,
            use_auxiliary=False,
            patch_size=patch_size,
        )
        if reinit_dpt_head:
            self._reinit_dpt_head()

    @property
    def lidar_prior(self):
        """Compatibility shim for diagnostics callback."""
        return self.decoder

    def _reinit_dpt_head(self) -> None:
        final_conv = self.head.scratch.output_conv2
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(
        self,
        image: torch.Tensor,
        chm: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_aux: bool = False,
    ):
        B, _, H, W = image.shape
        h_img = H // self.patch_size
        w_img = W // self.patch_size
        D = self.embed_dim

        feats = self.backbone(image)
        if len(feats) != 4:
            raise RuntimeError(
                f"Expected 4 backbone taps, got {len(feats)}."
            )

        if (
            chm is not None
            and self.training
            and torch.rand((), device=chm.device).item() < self.chm_dropout
        ):
            chm = None
        use_chm = chm is not None

        chm_tokens_with_pe: Optional[torch.Tensor] = None
        chm_recon: Optional[torch.Tensor] = None
        if use_chm:
            chm_tokens, h_chm, w_chm = self.chm_encoder(chm)

            if return_aux and self.chm_recon_head is not None:
                tokens_grid = (
                    chm_tokens.transpose(1, 2).reshape(B, D, h_chm, w_chm)
                )
                chm_recon = self.chm_recon_head(tokens_grid)
                if chm_recon.shape[-2:] != (H, W):
                    chm_recon = F.interpolate(
                        chm_recon, size=(H, W), mode="bilinear", align_corners=False,
                    )

            pe = build_2d_sincos_pe(
                h_chm, w_chm, D,
                device=chm_tokens.device, dtype=chm_tokens.dtype,
            )
            chm_tokens_with_pe = chm_tokens + pe.unsqueeze(0)

        chm_was_dropped = (
            torch.zeros(B, dtype=torch.bool, device=image.device)
            if use_chm
            else torch.ones(B, dtype=torch.bool, device=image.device)
        )

        stacked = torch.cat(feats, dim=0)
        chm_tokens_rep = (
            chm_tokens_with_pe.repeat(4, 1, 1)
            if chm_tokens_with_pe is not None
            else None
        )

        if return_intermediates:
            refined_stacked, per_layer, final_posterior_rep = self.decoder(
                stacked, chm_tokens_rep, None, return_intermediates=True,
            )
        else:
            refined_stacked = self.decoder(stacked, chm_tokens_rep, None)

        refined_list = list(torch.split(refined_stacked, B, dim=0))
        out_features = [(t,) for t in refined_list]
        height_map = self.head(out_features, h_img, w_img)

        intermediates = None
        if return_intermediates:
            final_posterior = final_posterior_rep[:B]
            intermediates = {
                "tap_indices": self.tap_indices,
                "img_tokens_per_tap": feats,
                "chm_tokens_with_pe": chm_tokens_with_pe,
                "chm_memory": final_posterior,
                "prior_bank": self.decoder.prior.detach(),
                "refined_tokens_per_tap": refined_list,
                "decoder_per_layer": per_layer,
            }
        aux = (
            {"chm_recon": chm_recon, "chm_was_dropped": chm_was_dropped}
            if return_aux
            else None
        )

        if return_intermediates and return_aux:
            return height_map, intermediates, aux
        if return_intermediates:
            return height_map, intermediates
        if return_aux:
            return height_map, aux
        return height_map

    def architecture_summary(self) -> str:
        def fmt_n(n: int) -> str:
            for unit in ("", "K", "M", "B"):
                if abs(n) < 1000:
                    return f"{n:.1f}{unit}" if unit else f"{n}"
                n /= 1000
            return f"{n:.1f}T"

        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        D = self.embed_dim
        M = self.num_prior_tokens
        taps = "+".join(str(i) for i in self.tap_indices)
        ls_desc = (
            "LayerScale: OFF (nn.Identity)"
            if self.layer_scale_init is None
            else f"LayerScale: ON  (γ init = {self.layer_scale_init:g})"
        )
        recon_line = (
            f"├── CHMReconHead (aux)                     chm_tokens → input-CHM recon (autoencoder)   params: {fmt_n(count(self.chm_recon_head))}"
            if self.chm_recon_head is not None else None
        )
        lines = [
            "Dinov3HeightModelV3DPT  (Plan v3 — per-layer prior refinement + DPT fusion)",
            f"├── Dinov3Backbone (frozen)                img → 4 taps at layers [{taps}]   params: {fmt_n(count(self.backbone))}",
            f"├── CHMPromptEncoder                       chm → chm_tokens [B, P', D={D}]   params: {fmt_n(count(self.chm_encoder))}",
        ]
        if recon_line is not None:
            lines.append(recon_line)
        lines += [
            f"├── HeightDecoderStackV3 (× {self.n_layers} layers, SHARED updater + SHARED across 4 taps)   params: {fmt_n(count(self.decoder))}",
            f"│   ├── prior    nn.Parameter [M={M}, D={D}]   ({fmt_n(M * D)} params)",
            f"│   ├── PosteriorUpdater (SHARED, called {self.n_layers}× per forward)   params: {fmt_n(count(self.decoder.updater))}",
            f"│   └── {self.n_layers} × HeightDecoderLayer   [{ls_desc}]",
            f"│       per layer: updater(posterior, chm) → CrossAttn(x, posterior) → SelfAttn → FFN",
            f"└── DPTHeadLinear (features={self.dpt_features})   4 taps → height_map   params: {fmt_n(count(self.head))}",
            "",
            f"Total params: {fmt_n(count(self))}   (backbone frozen)",
            f"Trainable    : {fmt_n(sum(p.numel() for p in self.parameters() if p.requires_grad))}",
        ]
        return "\n".join(lines)
