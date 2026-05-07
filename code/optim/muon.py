"""Muon optimiser hybrid wrapper (upstream-backed).

Reference:
    Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"
    https://kellerjordan.github.io/posts/muon/
    Vendored into PyTorch 2.11+ at ``torch/optim/_muon.py``.

Why Muon for the U-series
-------------------------
Muon's per-step update for a 2-D weight ``W`` is the *Newton-Schulz*
projection of the momentum-buffer onto the orthogonal manifold,
followed by a width-aware spectral scaling. The spectral norm of every
Muon update is therefore exactly ``lr * max(1, sqrt(d_out/d_in))`` —
*independent of gradient magnitude*. Three consequences directly
address the failures we hit in K7 / T1 / T10 / T11:

1. **No more spectral collapse from the optimiser side.** AdamW
   normalises *element-wise* (per-coordinate ``g/sqrt(v̂)``), which
   leaves the update free to concentrate all of its energy in the top
   singular direction of the gradient — the σReparam paper's
   Proposition 3.1. Muon's update has its full singular spectrum
   constrained to ``≈ 1``, so the *update direction* never becomes
   rank-1 even when the gradient does.
2. **σReparam-friendly.** Each Muon step changes the weight by a fixed
   spectral norm ``≈ lr`` (see eq. 4 in the post). This is exactly the
   per-step bound the σReparam ``γ`` learnable scalar wants to track —
   the two interventions reinforce rather than fight each other.
3. **Width-independent dynamics.** ``max(1, sqrt(d_out/d_in))`` rescales
   for rectangular weights so that 2-D matrices of different shapes
   still receive comparable updates. Empirically this is what enables
   the larger LR (``2e-2`` ~ 5×–10× AdamW's typical) without
   instability.

Hybrid layout
-------------
We use Muon for the 2-D matrices of the trainable surface (decoder Q,
K, V, output projections, FFN weights, DPT-head linears) and AdamW for
everything else (LayerNorm γ/β, biases, the prior bank, σReparam γ
scalars, gate parameters, 3-D / 4-D conv weights). This is the
canonical recipe from Keller's reference repo and the upstream
PyTorch docs.

Implementation
--------------
This wrapper now delegates the actual Newton-Schulz iteration to the
upstream ``torch.optim._muon.muon`` functional API (vendored in
PyTorch 2.11). We keep a thin ``Optimizer`` subclass around it so:

* Lightning sees a single optimiser object — one ``step()``,
  one ``zero_grad()``, a flat ``param_groups`` for LR schedulers and
  gradient clipping.
* The Muon group sits next to one or two AdamW groups (decay /
  no-decay) in the same optimiser. The dispatcher checks the
  ``use_muon`` flag on each group and routes to either
  :func:`torch.optim._muon.muon` or our local AdamW kernel.
* Optimiser state is stored in ``self.state[p]`` per parameter, so
  standard PyTorch / Lightning checkpoint plumbing just works.

The local AdamW kernel is intentionally retained (rather than going
through ``torch.optim.AdamW``) so that switching the Muon kernel is
the *only* behavioural change vs. the original vendored Muon — the
AdamW group's numerics stay bit-identical to prior U-series runs.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch.optim._muon import muon as _muon_functional
from torch.optim.optimizer import Optimizer


_DEFAULT_NS_COEFFICIENTS: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
_DEFAULT_EPS: float = 1e-7


class Muon(Optimizer):
    """Muon + AdamW hybrid optimiser (thin wrapper around upstream Muon).

    Param groups are dispatched via the ``use_muon`` flag set per group
    in the constructor:

    * ``use_muon=True`` group → upstream ``torch.optim._muon.muon``
      functional API. 2-D weights only.
    * ``use_muon=False`` group → local decoupled-weight-decay AdamW
      kernel (kept identical to the previous vendored implementation
      for run-to-run comparability).

    Args:
        muon_params: iterable of 2-D parameters to update with Muon.
        lr: Muon learning rate. Used as the per-step spectral norm of
            the update (modulo the ``adjust_lr_fn`` shape factor).
            Suggested ``2e-2`` for our 1024-dim decoder; this is
            intentionally larger than AdamW's typical ``1e-4`` since
            the orthogonalised update is dimensionally independent.
        momentum: SGD momentum coefficient for the Muon group.
        nesterov: whether to use Nesterov momentum.
        ns_steps: Newton-Schulz iteration count. ``5`` matches Keller's
            reference; enough for training stability.
        ns_coefficients: ``(a, b, c)`` for the NS quintic. Defaults to
            Keller's tuned values ``(3.4445, -4.7750, 2.0315)``.
        eps: epsilon for the unit-spectral-norm prescaling inside NS.
        adjust_lr_fn: width-aware LR scaling. ``"original"`` (default,
            Keller) → ``sqrt(max(1, A/B))``. ``"match_rms_adamw"``
            (Moonshot's "Muon is Scalable for LLM Training", Feb 2025)
            → ``0.2 * sqrt(max(A, B))`` — lets you reuse AdamW-tuned
            LR/WD directly. ``None`` → no width scaling.
        adamw_params: iterable of parameters to update with AdamW. May
            be ``None`` (Muon-only).
        adamw_lr: AdamW learning rate. Decoupled from ``lr`` because
            the units of "good LR" are different for the two groups.
        adamw_betas: AdamW ``(β1, β2)``.
        adamw_eps: AdamW ε.
        adamw_wd: AdamW weight decay (decoupled from LR, AdamW-style).
        adamw_no_decay_params: optional iterable of AdamW params that
            should *not* receive weight decay (LayerNorm γ/β, biases,
            the prior bank, σReparam γ scalars, gate parameters …).
            When ``None``, the single AdamW group with weight decay
            ``adamw_wd`` is used for everything.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: Tuple[float, float, float] = _DEFAULT_NS_COEFFICIENTS,
        eps: float = _DEFAULT_EPS,
        adjust_lr_fn: Optional[str] = "original",
        adamw_params: Optional[Iterable[torch.nn.Parameter]] = None,
        adamw_lr: float = 1e-4,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
        adamw_no_decay_params: Optional[Iterable[torch.nn.Parameter]] = None,
    ):
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(
                f"adjust_lr_fn must be one of 'original', 'match_rms_adamw', or None; "
                f"got {adjust_lr_fn!r}"
            )

        muon_params = list(muon_params)
        for i, p in enumerate(muon_params):
            if p.ndim != 2:
                raise ValueError(
                    f"Muon expects 2D parameters; muon_params[{i}] has shape "
                    f"{tuple(p.shape)} (ndim={p.ndim}). Move it to adamw_params."
                )

        param_groups: List[dict] = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": float(lr),
                "momentum": float(momentum),
                "nesterov": bool(nesterov),
                "ns_steps": int(ns_steps),
                "ns_coefficients": tuple(float(c) for c in ns_coefficients),
                "eps": float(eps),
                "adjust_lr_fn": adjust_lr_fn,
                "weight_decay": 0.0,
            }
        ]

        if adamw_params is not None:
            param_groups.append(
                {
                    "params": list(adamw_params),
                    "use_muon": False,
                    "lr": float(adamw_lr),
                    "betas": tuple(adamw_betas),
                    "eps": float(adamw_eps),
                    "weight_decay": float(adamw_wd),
                }
            )

        if adamw_no_decay_params is not None:
            param_groups.append(
                {
                    "params": list(adamw_no_decay_params),
                    "use_muon": False,
                    "lr": float(adamw_lr),
                    "betas": tuple(adamw_betas),
                    "eps": float(adamw_eps),
                    "weight_decay": 0.0,
                }
            )

        # ``defaults`` is the per-key fallback Optimizer consults for any
        # group key that wasn't set explicitly. Every key we care about
        # is set per-group above, so ``defaults`` is just a backstop.
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            ns_coefficients=tuple(float(c) for c in ns_coefficients),
            eps=float(eps),
            adjust_lr_fn=adjust_lr_fn,
            betas=tuple(adamw_betas),
            weight_decay=float(adamw_wd),
            use_muon=True,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    # ------------------------------------------------------------------
    # Muon update (delegates to upstream torch.optim._muon.muon)
    # ------------------------------------------------------------------
    def _muon_step(self, group: dict) -> None:
        params: List[torch.Tensor] = []
        grads: List[torch.Tensor] = []
        muon_momentum_bufs: List[torch.Tensor] = []

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            params.append(p)
            grads.append(p.grad)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            muon_momentum_bufs.append(state["momentum_buffer"])

        if not params:
            return

        # Single call into upstream functional API. weight_decay is
        # always 0 for the Muon group in our hybrid layout (decay lives
        # on the AdamW group instead, matching Keller's reference recipe).
        _muon_functional(
            params,
            grads,
            muon_momentum_bufs,
            foreach=None,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            ns_coefficients=group["ns_coefficients"],
            eps=group["eps"],
            ns_steps=group["ns_steps"],
            adjust_lr_fn=group["adjust_lr_fn"],
            has_complex=False,
        )

    # ------------------------------------------------------------------
    # AdamW update (decoupled weight decay) — kept local on purpose so
    # the only behavioural change vs. the previous vendored Muon is the
    # NS kernel itself. AdamW numerics stay bit-identical to prior runs.
    # ------------------------------------------------------------------
    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
            step_size = lr / bias_correction1

            if wd > 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-step_size)


def split_params_for_muon(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    no_decay_predicate=None,
) -> dict:
    """Split ``model.named_parameters()`` into the three Muon param sets.

    Returns a dict with keys ``muon`` / ``adamw_decay`` / ``adamw_no_decay``.

    Splitting rules:

    * **muon**: ``param.ndim == 2`` and *not* a vector-shaped weight
      (``shape[0] >= 2`` and ``shape[1] >= 2``). All decoder Q, K, V,
      output projections, FFN linears, DPT-head linears land here.
    * **adamw_no_decay**: any of the following — the rest gets weight
      decay:
        - ``param.ndim <= 1`` (LayerNorm γ/β, biases, σReparam ``γ``
          scalars, ``alpha`` / ``post_ln_alpha`` gates, GRU bias slices,
          ``layerscale`` ``γ`` vector).
        - parameter name contains ``"prior"`` (the prior bank — kept in
          ``no_decay`` exactly as the existing ``HeightEstimationModule.
          configure_optimizers`` does).
        - parameter name contains ``"slot_embed"`` (DETR-style slot
          identity embedding — same rationale as the prior bank: it's a
          learned per-slot positional/identity tensor and weight decay
          on it would shrink the symmetry-breaking signal toward zero).
        - ``no_decay_predicate(name, param) == True`` if the caller
          provides a custom predicate.
    * **adamw_decay**: everything else (3-D / 4-D conv weights, e.g.
      ``CHMPromptEncoder.conv``).

    Why DPT-head linears go to Muon: they are ``Linear(D → C)`` 2-D
    weights and benefit from the same spectral-conditioning argument
    as the decoder. The DPT *fusion conv* tensors are 4-D and end up in
    ``adamw_decay`` since Muon is defined only on 2-D matrices.
    """
    muon: List[torch.nn.Parameter] = []
    adamw_decay: List[torch.nn.Parameter] = []
    adamw_no_decay: List[torch.nn.Parameter] = []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        is_no_decay = (
            param.ndim <= 1
            or "prior" in name
            or "slot_embed" in name
            or (no_decay_predicate is not None and no_decay_predicate(name, param))
        )
        if is_no_decay:
            adamw_no_decay.append(param)
        elif param.ndim == 2 and min(param.shape) >= 2:
            muon.append(param)
        else:
            adamw_decay.append(param)

    return {
        "muon": muon,
        "adamw_decay": adamw_decay,
        "adamw_no_decay": adamw_no_decay,
    }
