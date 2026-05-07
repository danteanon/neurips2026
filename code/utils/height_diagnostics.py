"""CHM diagnostics: Track A probes P1-P5 + attention-capture hooks.

Single source of truth for:
  * Attention weight capture (``AttentionCollector`` context manager)
  * MHA module discovery (``find_mha_modules``)
  * Probe implementations (``probe_p1_entropy`` ... ``probe_p5_reliance``)
  * Derived health flags (``compute_health_flags``)
  * Threshold table (``THRESHOLDS``)

Used by two consumers:

1. In-training: ``lightning_modules.chm_diagnostics_callback.CHMDiagnosticsCallback``
   runs every validation epoch and streams scalars / histograms / images to
   TensorBoard under the ``diag/`` namespace. This is how we monitor while
   training and decide whether to kill a run early.
2. Post-training: ``scripts/chm/run_diagnostics.py`` runs the same probes on
   ``n_samples`` validation examples and writes per-probe CSVs + PNGs + a
   ``summary.json`` to ``docs/chm/diagnostics/<run_id>/``.

Probe coverage matrix::

    P1 (attention entropy)          — all models (v1 + v2)
    P2 (V-projection norms)         — all models
    P3 (posterior-prior delta)      — Plan v2 only (needs ``LidarPriorLayer``)
    P4 (slot coverage)              — Plan v2 only
    P5 (prior-vs-evidence reliance) — Plan v2 with ``concat_chm_to_memory=True`` only

Design rationale for recomputing attention from hook-cached inputs:
  The custom :class:`model.dinov3_height_model.MultiHeadAttention` discards the
  post-softmax attention tensor immediately after computing the weighted sum,
  so we cannot read it from the forward output. Instead we register a single
  forward hook per MHA that caches ``q_source`` / ``kv_source`` / ``attn_mask``,
  then recompute :math:`\\mathrm{softmax}(QK^\\top / \\sqrt{d})` on demand. The
  recompute is cheap (a few matmuls) and deterministic because the MHA is
  eval-mode with dropout disabled when the collector is active.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Thresholds — single source of truth for both the live callback and the
# post-training script. Tune here once; downstream consumers import them.
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, float] = {
    # P2 — prior bank slot health
    "dead_slot_norm_frac": 0.1,        # slot dead if ‖s‖ < frac * mean(‖s‖)

    # P3 — posterior-prior delta per slot
    "dormant_slot_delta": 0.05,        # slot dormant (CHM not moving it) if δ < this
    "collapse_slot_delta": 2.0,        # slot collapsed (wiped by CHM) if δ > this

    # P4 — slot coverage (specialisation)
    "slot_specialized_norm_H": 0.5,    # slot "specialized" if H / log(P') < this

    # Health flag thresholds
    "prior_collapse_mean_delta": 2.0,    # prior_collapse_risk if P3.mean_delta > this
    "image_agnostic_mean_delta": 0.05,   # image_agnostic_risk if P3.mean_delta < this
    "slot_collapse_coverage_ratio": 0.3, # slot_collapse_risk: P4.coverage_ratio < this ...
    "slot_collapse_updater_H": 1.0,      # ... AND P1 updater entropy < this

    # Post-training regime comparison
    "regime_insensitive_P3_std": 0.02,   # regime_insensitive if std(P3) across regimes < this ...
    "regime_insensitive_P1_std": 0.05,   # ... AND std(P1 decoder entropy) across regimes < this
    "chm_decorative_mae_std": 0.1,       # chm_decorative adds std(MAE) across regimes < this

    # Spectral collapse of decoder cross-attn v_proj (K5/T0/K6/T1 post-mortem)
    # A "collapsed" layer has its top singular value carry most of the Frobenius
    # mass (stable_rank close to 1) and its entropy-based effective rank well
    # below half of min(d_out, d_in). Live-training flag fires when either
    # sigma_max tripled vs init or effective_rank halved below 50% of init.
    "spectral_collapse_stable_rank": 4.0,     # flag if stable_rank < this
    "spectral_collapse_effrank_frac": 0.3,    # flag if effective_rank / min_dim < this
    "spectral_collapse_max_attn": 0.8,        # one-hot attn: mean(max_row) > this
}


# ---------------------------------------------------------------------------
# Module discovery — works for Plan v1 (no prior) and Plan v2 (with prior).
# ---------------------------------------------------------------------------

def find_mha_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Return ``{qualified_name: MultiHeadAttention}`` for every MHA in the model.

    Qualified names (stable across epochs, usable as TB tag suffixes)::

        decoder.layers.<i>.cross_attn.mha     for i in 0 .. n_layers-1
        decoder.layers.<i>.self_attn.mha      for i in 0 .. n_layers-1
        lidar_prior.updater.cross_attn        (Plan v2 only)
    """
    out: Dict[str, nn.Module] = {}
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        for i, layer in enumerate(decoder.layers):
            cross = getattr(layer, "cross_attn", None)
            if cross is not None and hasattr(cross, "mha"):
                out[f"decoder.layers.{i}.cross_attn.mha"] = cross.mha
            self_ = getattr(layer, "self_attn", None)
            if self_ is not None and hasattr(self_, "mha"):
                out[f"decoder.layers.{i}.self_attn.mha"] = self_.mha

    lidar_prior = getattr(model, "lidar_prior", None)
    if lidar_prior is not None:
        updater = getattr(lidar_prior, "updater", None)
        if updater is not None and hasattr(updater, "cross_attn"):
            out["lidar_prior.updater.cross_attn"] = updater.cross_attn
    return out


def _token_kind(name: str) -> str:
    """Infer the semantic role of the KV stream feeding an MHA.

    - ``'img'``: decoder self-attn (image tokens queried against themselves).
    - ``'memory'``: decoder cross-attn (memory is CHM tokens for Plan v1 or
      the posterior bank for Plan v2; we keep one label because the decoder
      doesn't know which it is getting).
    - ``'chm_evidence'``: Plan v2 posterior updater (prior queries CHM tokens).
    """
    if "lidar_prior.updater" in name:
        return "chm_evidence"
    if "self_attn" in name:
        return "img"
    if "cross_attn" in name:
        return "memory"
    return "unknown"


# ---------------------------------------------------------------------------
# Attention collector — captures the MHA inputs so we can recompute attention
# weights without changing model code.
# ---------------------------------------------------------------------------

class AttentionCollector:
    """Context manager that captures MHA inputs and recomputes attention weights.

    Usage::

        mhas = find_mha_modules(model)
        with AttentionCollector(mhas) as coll:
            out = model(image, chm, return_intermediates=True)
        for name in coll.captured_names:
            attn = coll.attn(name)     # [B, h, T_q, T_kv] post-softmax

    The collector stores a per-module cache of ``(q_source, kv_source,
    attn_mask)`` and exposes lazy helpers that recompute attention weights
    and V-projection norms on demand. Compute cost: one extra ``q_proj`` +
    ``k_proj`` matmul + softmax per attention query. Cheap.

    Safety:
      * Hooks are removed automatically on ``__exit__``, even when the
        forward pass raises.
      * Inputs are detached but stay on the original device.
      * Attention mask (``True = blocked``) is preserved and re-applied by
        ``attn()`` so diagnostics match the real runtime softmax.
    """

    def __init__(self, mha_modules: Dict[str, nn.Module]):
        self._mha_modules = dict(mha_modules)
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._handles: list = []

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "AttentionCollector":
        self._cache.clear()
        for name, mha in self._mha_modules.items():
            handle = mha.register_forward_hook(self._make_hook(name), with_kwargs=True)
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------
    def _make_hook(self, name: str):
        def hook(module, args, kwargs, output):
            q_source = args[0] if len(args) > 0 else kwargs.get("q_source")
            kv_source = args[1] if len(args) > 1 else kwargs.get("kv_source")
            if len(args) > 2:
                attn_mask = args[2]
            else:
                attn_mask = kwargs.get("attn_mask")
            self._cache[name] = {
                "q_source": q_source.detach(),
                "kv_source": kv_source.detach(),
                "attn_mask": attn_mask.detach() if torch.is_tensor(attn_mask) else attn_mask,
            }
        return hook

    # ------------------------------------------------------------------
    # Lazy recomputation
    # ------------------------------------------------------------------
    @property
    def captured_names(self) -> List[str]:
        return list(self._cache.keys())

    def attn(self, name: str) -> torch.Tensor:
        """``[B, h, T_q, T_kv]`` post-softmax attention for one MHA.

        Uses the MHA's public ``project_q`` / ``project_k`` helpers so it
        works for both fused-self-attn (``Linear(D, 3D)`` ``qkv_proj``) and
        separate-Q + fused-KV cross-attn layouts. If the MHA has
        ``qk_norm=True`` the recomputation also passes Q and K through the
        per-head LayerNorms so the attention pattern matches what the
        forward pass actually produced.
        """
        entry = self._cache[name]
        mha = self._mha_modules[name]
        q_source = entry["q_source"]
        kv_source = entry["kv_source"]
        attn_mask = entry["attn_mask"]

        B, T_q, _ = q_source.shape
        T_kv = kv_source.shape[1]
        h, d_h = mha.n_heads, mha.d_head

        Q = mha.project_q(q_source).view(B, T_q, h, d_h).transpose(1, 2)
        K = mha.project_k(kv_source).view(B, T_kv, h, d_h).transpose(1, 2)
        if getattr(mha, "qk_norm", False):
            Q = mha.q_ln(Q)
            K = mha.k_ln(K)
        logits = (Q @ K.transpose(-2, -1)) / mha.scale
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask, float("-inf"))
        return logits.softmax(dim=-1)

    def v_norm(self, name: str) -> torch.Tensor:
        """``[B, T_kv]`` L2 norms of the V-projected tokens."""
        entry = self._cache[name]
        mha = self._mha_modules[name]
        V = mha.project_v(entry["kv_source"])
        return V.norm(dim=-1)

    def kv_norm(self, name: str) -> torch.Tensor:
        """``[B, T_kv]`` L2 norms of the post-LN kv_source (what the MHA sees)."""
        return self._cache[name]["kv_source"].norm(dim=-1)


# ---------------------------------------------------------------------------
# P1 — attention entropy
# ---------------------------------------------------------------------------

def _entropy_per_query(attn: torch.Tensor) -> torch.Tensor:
    """Shannon entropy along the last axis.

    attn: ``[..., T_kv]`` post-softmax (rows sum to 1).
    Returns: same shape minus the last dim.
    """
    eps = 1e-12
    p = attn.clamp(min=0.0)
    return -(p * (p + eps).log()).sum(dim=-1)


def probe_p1_entropy(collector: AttentionCollector) -> Dict[str, Dict[str, float]]:
    """P1 — per-layer attention entropy.

    For every captured MHA, returns ``mean_H`` (raw nats), ``mean_H_norm``
    (divided by log(T_kv), so :math:`\\in [0, 1]` for unmasked attention),
    and dispersion stats.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in collector.captured_names:
        attn = collector.attn(name)                        # [B, h, T_q, T_kv]
        T_kv = attn.shape[-1]
        H = _entropy_per_query(attn).reshape(-1).detach()  # flatten across b, h, q
        max_H = math.log(max(T_kv, 2))
        out[name] = {
            "layer_kind": _layer_kind(name),
            "token_kind": _token_kind(name),
            "mean_H": float(H.mean()),
            "std_H": float(H.std()),
            "min_H": float(H.min()),
            "max_H": float(H.max()),
            "mean_H_norm": float(H.mean() / max_H),
            "max_possible_H": max_H,
            "T_kv": int(T_kv),
        }
    return out


def _layer_kind(name: str) -> str:
    if "cross_attn" in name:
        return "cross"
    if "self_attn" in name:
        return "self"
    return "cross"  # updater is cross-attn


def summarise_p1(p1: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-layer P1 stats into scalars suitable for TB."""
    def _mean_over(filter_fn):
        vals = [per["mean_H"] for name, per in p1.items() if filter_fn(name, per)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "decoder_xattn_mean_H": _mean_over(
            lambda n, p: n.startswith("decoder.layers.") and "cross_attn" in n
        ),
        "decoder_self_mean_H": _mean_over(
            lambda n, p: n.startswith("decoder.layers.") and "self_attn" in n
        ),
        "updater_xattn_mean_H": _mean_over(
            lambda n, p: "lidar_prior.updater" in n
        ),
        "decoder_xattn_mean_H_norm": (
            sum(
                per["mean_H_norm"]
                for n, per in p1.items()
                if n.startswith("decoder.layers.") and "cross_attn" in n
            ) / max(
                sum(
                    1 for n in p1 if n.startswith("decoder.layers.") and "cross_attn" in n
                ),
                1,
            )
        ),
    }


# ---------------------------------------------------------------------------
# P2 — V-projection norms
# ---------------------------------------------------------------------------

def probe_p2_norms(collector: AttentionCollector) -> Dict[str, Dict[str, float]]:
    """P2 — per-layer V-projection norm statistics.

    For every captured MHA we record mean / quartiles of
    :math:`\\lVert V(\\mathrm{LN}(kv))\\rVert_2` per token, plus the raw
    post-LN ``kv_source`` norm. The V-projected norm is the "voice volume"
    the softmax downstream will weight-average; the raw KV norm is what
    LayerNorm hands the MHA.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in collector.captured_names:
        v_norm = collector.v_norm(name).reshape(-1).detach()
        kv_norm = collector.kv_norm(name).reshape(-1).detach()

        q_vals = torch.tensor([0.25, 0.5, 0.75], device=v_norm.device)
        q25, q50, q75 = torch.quantile(v_norm, q_vals).tolist()

        out[name] = {
            "layer_kind": _layer_kind(name),
            "token_kind": _token_kind(name),
            "mean_vnorm": float(v_norm.mean()),
            "std_vnorm": float(v_norm.std()),
            "q25_vnorm": q25,
            "q50_vnorm": q50,
            "q75_vnorm": q75,
            "mean_kvnorm": float(kv_norm.mean()),
            "std_kvnorm": float(kv_norm.std()),
        }
    return out


def summarise_p2(p2: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-layer P2 stats into per-kind scalars."""
    def _mean_by_kind(kind: str) -> float:
        vals = [per["mean_vnorm"] for per in p2.values() if per["token_kind"] == kind]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "img_vnorm_mean": _mean_by_kind("img"),
        "memory_vnorm_mean": _mean_by_kind("memory"),
        "chm_evidence_vnorm_mean": _mean_by_kind("chm_evidence"),
    }


def probe_prior_bank_health(
    prior_bank: torch.Tensor,
    dead_frac: float = THRESHOLDS["dead_slot_norm_frac"],
) -> Dict[str, float]:
    """Prior-bank-level P2 adjunct (v2 only): slot norm statistics + dead slots.

    A slot is "dead" if its L2 norm is below ``dead_frac * mean(all slot norms)``.
    This is a simple scale-free check that flags slots the optimiser has left
    near their random init after training.
    """
    norms = prior_bank.detach().norm(dim=-1)               # [M]
    mean_norm = float(norms.mean())
    dead_mask = norms < dead_frac * mean_norm if mean_norm > 0 else torch.zeros_like(norms, dtype=torch.bool)
    return {
        "mean_slot_norm": mean_norm,
        "min_slot_norm": float(norms.min()),
        "max_slot_norm": float(norms.max()),
        "std_slot_norm": float(norms.std()),
        "dead_slot_fraction": float(dead_mask.float().mean()),
        "dead_slot_count": int(dead_mask.sum()),
        "num_slots": int(norms.numel()),
    }


# ---------------------------------------------------------------------------
# P3 — posterior-prior delta (Plan v2 only)
# ---------------------------------------------------------------------------

def probe_p3_delta(
    prior_bank: torch.Tensor,
    chm_memory: torch.Tensor,
    num_prior_tokens: int,
    concat_chm_to_memory: bool,
    dormant_thresh: float = THRESHOLDS["dormant_slot_delta"],
    collapse_thresh: float = THRESHOLDS["collapse_slot_delta"],
) -> Dict[str, object]:
    """P3 — posterior-prior delta.

    Args:
        prior_bank: ``[M, D]`` — ``model.lidar_prior.prior`` (detached).
        chm_memory: ``[B, M, D]`` when pure-B, ``[B, M+P', D]`` when concat.
        num_prior_tokens: ``M``.
        concat_chm_to_memory: whether the memory carries concatenated CHM tokens.

    Returns:
        dict with headline ``mean_delta`` etc. plus the per-slot
        ``delta_per_slot`` tensor suitable for a TB histogram.
    """
    prior_bank = prior_bank.detach()
    chm_memory = chm_memory.detach()
    if concat_chm_to_memory:
        posterior = chm_memory[:, :num_prior_tokens, :]
    else:
        posterior = chm_memory[:, :num_prior_tokens, :]    # matches pure-B too

    prior = prior_bank.unsqueeze(0).expand_as(posterior)
    num = (posterior - prior).norm(dim=-1)                 # [B, M]
    den = prior.norm(dim=-1).clamp(min=1e-6)
    delta = num / den                                       # [B, M]
    delta_per_slot = delta.mean(dim=0)                     # [M]

    dormant = (delta_per_slot < dormant_thresh)
    collapse = (delta_per_slot > collapse_thresh)
    M = delta_per_slot.numel()

    return {
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std()),
        "min_delta": float(delta.min()),
        "max_delta": float(delta.max()),
        "dormant_slots": int(dormant.sum()),
        "dormant_fraction": float(dormant.float().mean()),
        "collapse_slots": int(collapse.sum()),
        "collapse_fraction": float(collapse.float().mean()),
        "num_slots": M,
        "delta_per_slot": delta_per_slot.detach().cpu(),
    }


# ---------------------------------------------------------------------------
# P4 — slot coverage (Plan v2 only)
# ---------------------------------------------------------------------------

def probe_p4_coverage(
    updater_attn: torch.Tensor,
    slot_norm_H_thresh: float = THRESHOLDS["slot_specialized_norm_H"],
) -> Dict[str, object]:
    """P4 — prior-slot coverage of CHM evidence.

    Args:
        updater_attn: ``[B, h, M, P']`` post-softmax attention from the
            :class:`LidarPriorLayer` updater.
        slot_norm_H_thresh: a slot is "specialized" if its normalized
            entropy falls below this threshold. Lower values require tighter
            peaks before a slot counts as specialized.

    Returns:
        Dict with scalar metrics + an ``attn_avg`` tensor ``[M, P']`` for
        the slot-attention heatmap.

    ``coverage_ratio`` is the fraction of slots that are specialized
    (peaked). Low ``coverage_ratio`` with low overall updater entropy means a
    few slots monopolise the CHM signal — slot collapse risk.
    """
    updater_attn = updater_attn.detach()
    B, h, M, P_prime = updater_attn.shape
    p_slot = updater_attn.mean(dim=1)                       # [B, M, P']
    p_slot = p_slot / p_slot.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    eps = 1e-12
    H_per_slot = -(p_slot * (p_slot + eps).log()).sum(dim=-1)   # [B, M]
    max_H = math.log(max(P_prime, 2))
    H_norm = H_per_slot / max_H                             # [B, M] in [0, 1]

    specialized = (H_norm < slot_norm_H_thresh).float()
    coverage_ratio = float(specialized.mean())

    return {
        "mean_per_slot_entropy": float(H_per_slot.mean()),
        "mean_per_slot_entropy_norm": float(H_norm.mean()),
        "max_per_slot_entropy": float(H_per_slot.max()),
        "max_possible_entropy": max_H,
        "coverage_ratio": coverage_ratio,
        "num_slots": M,
        "num_keys": P_prime,
        "attn_avg": p_slot.mean(dim=0).detach().cpu(),       # [M, P']
    }


# ---------------------------------------------------------------------------
# P5 — prior-vs-evidence reliance (Plan v2 with concat only)
# ---------------------------------------------------------------------------

def probe_p5_reliance(
    decoder_xattn_per_layer: List[torch.Tensor],
    num_prior_tokens: int,
    h_img: int,
    w_img: int,
) -> Dict[str, object]:
    """P5 — per-pixel reliance on prior vs raw CHM evidence.

    Meaningful only when ``concat_chm_to_memory=True`` so the decoder sees
    memory laid out as ``[first M prior keys | next P' CHM evidence keys]``.

    For each decoder cross-attention layer, split the per-query softmax mass
    into "prior" (first M keys) and "evidence" (remaining keys), average
    over heads and layers, and reshape per-query mass to the image grid.

    Args:
        decoder_xattn_per_layer: one ``[B, h, P, M+P']`` attention tensor
            per decoder layer.
        num_prior_tokens: ``M``.
        h_img, w_img: image token grid shape.

    Returns:
        Dict with scalar reliance + a ``reliance_map [B, h_img, w_img]``
        suitable for TB image logging. High values = decoder relied on the
        prior at that spatial location; low values = decoder relied on raw
        CHM evidence.
    """
    per_layer_prior_mass = []
    for attn in decoder_xattn_per_layer:
        attn = attn.detach()
        prior_mass = attn[..., :num_prior_tokens].sum(dim=-1)     # [B, h, P]
        per_layer_prior_mass.append(prior_mass.mean(dim=1))       # [B, P]
    prior_mass = torch.stack(per_layer_prior_mass, dim=0).mean(dim=0)  # [B, P]

    B, P = prior_mass.shape
    if P != h_img * w_img:
        raise ValueError(
            f"P5: expected P = h_img*w_img ({h_img}*{w_img} = {h_img*w_img}), got P={P}"
        )
    reliance_map = prior_mass.reshape(B, h_img, w_img)

    return {
        "mean_prior_reliance": float(prior_mass.mean()),
        "median_prior_reliance": float(prior_mass.median()),
        "min_prior_reliance": float(prior_mass.min()),
        "max_prior_reliance": float(prior_mass.max()),
        "reliance_map": reliance_map.detach().cpu(),              # [B, h_img, w_img]
    }


# ---------------------------------------------------------------------------
# CHM encoder output probe (I1).
#
# Upstream complement to the decoder W_v spectral probe: instead of asking
# "is the decoder concentrating its V-projection?", this asks "is the CHM
# encoder producing feature-rich, diverse tokens?" If the CHM encoder's
# output is already low-rank or dead-token-heavy, no amount of decoder
# regularisation will recover the information. The T4 P4=0.016 finding
# motivated adding this probe: we need to know whether the bottleneck is
# "prior bank ignores rich CHM signal" or "CHM signal is already impoverished
# before it reaches the prior bank".
# ---------------------------------------------------------------------------


def probe_chm_encoder_stats(
    chm_tokens: torch.Tensor,
    dead_token_norm_thresh: float = 0.05,
) -> Dict[str, float]:
    """CHM encoder output statistics.

    Args:
        chm_tokens: ``[B, P', D]`` tokens from :class:`CHMPromptEncoder`
            (either raw or with positional encoding added — statistics are
            relative enough to be meaningful on either).
        dead_token_norm_thresh: tokens with L2 norm below this are counted
            as "dead". Default 0.05 which is ~7 % of a healthy init-scale
            LayerNorm output; well below anything a forward pass would
            produce organically.

    Returns:
        Dict with:
          * ``token_norm_mean`` / ``token_norm_std`` — distribution of L2
            norms across the ``[B, P']`` tokens.
          * ``token_norm_min`` / ``token_norm_max``.
          * ``dead_token_fraction`` — ``mean(norm < thresh)``. A healthy
            run stays at 0. Rising dead fraction means the encoder is
            producing zero vectors for some spatial locations.
          * ``spatial_variance_mean`` — mean variance across the batch
            axis for each spatial location. Measures whether the encoder
            produces location-sensitive features (spatial_variance_mean
            near 0 = encoder collapses to a per-location constant
            regardless of input CHM content).
          * ``mean_to_token_ratio`` — ``||mean_token|| /
            mean(||token_i||)``. Direct measure of "all tokens cluster
            around the same direction": ratio → 1 means every token is
            essentially the bank-mean (severe DC collapse); ratio → 0
            means tokens disperse around zero.
          * ``cos_offdiag_mean`` — mean off-diagonal pairwise cosine of
            L2-normalised tokens (computed on a 512-token subsample to
            keep the cost bounded). Cosine → 1 means all tokens point in
            the same direction (rank-1 bank).
          * ``sigma_max`` / ``frob`` / ``effective_rank_raw`` /
            ``stable_rank_raw`` — spectral health of the *uncentered*
            ``[B*P', D]`` flat token matrix. Captures the full DC
            structure of the bank; ``stable_rank_raw → 1`` is the strict
            "every token lies on a single line" pathology.
          * ``effective_rank_centered`` / ``stable_rank_centered`` — same
            but on ``flat - mean(flat)``. Captures *residual* directional
            diversity once the shared-mean direction is removed; useful
            in conjunction with the raw versions to tell "rank-1 because
            everything is the mean" from "rank-1 because the residual is
            also collapsed".

    Cost: two SVDs of ``[B*P', D]`` plus a ``[512, 512]`` cosine matrix
    (≈25 ms at B=4, P'=256, D=1024). Safe for every tier-B probe
    (500 steps) and every val epoch.
    """
    t = chm_tokens.detach().to(torch.float32)
    B, P_prime, D = t.shape

    # --- Per-token norm distribution -------------------------------------
    token_norms = t.norm(dim=-1)                              # [B, P']
    token_norm_mean_val = float(token_norms.mean())
    dead_token_fraction = float((token_norms < dead_token_norm_thresh).float().mean())

    # --- Spatial variance -------------------------------------------------
    # Per-spatial-location variance across batch of each feature channel,
    # then averaged across channels and spatial locations. Near-zero value
    # means the encoder ignores its input and outputs the same features at
    # each spatial location regardless of the CHM content.
    spatial_variance_mean = float(t.var(dim=0, unbiased=False).mean())

    # --- Spectral health of the flat [B*P', D] token matrix --------------
    # Two views: raw (captures DC collapse — all tokens at the same point)
    # and centred (captures residual directional diversity around the
    # mean). The earlier callback only logged the centred view, which
    # silently hides the dominant pathology when every token clusters
    # near one location in feature space.
    flat = t.reshape(B * P_prime, D)
    spec_raw = weight_spectral_health(flat)
    flat_centred = flat - flat.mean(dim=0, keepdim=True)
    spec_centred = weight_spectral_health(flat_centred)

    # --- Mean-to-token-norm ratio (DC collapse indicator) ---------------
    mean_token = flat.mean(dim=0)                             # [D]
    mean_token_norm = float(mean_token.norm())
    mean_to_token_ratio = mean_token_norm / max(token_norm_mean_val, 1e-12)

    # --- Pairwise off-diagonal cosine on a bounded subsample ------------
    # Cap the cosine matrix at 512x512 to keep cost stable independent of
    # B*P_prime. With B=4, P'=256 we have 1024 tokens; pick the first 512
    # in fixed order so the metric is reproducible across runs.
    cos_n = min(int(B * P_prime), 512)
    sub = flat[:cos_n]
    sub_n = sub / sub.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    G = sub_n @ sub_n.T                                       # [cos_n, cos_n]
    eye = torch.eye(cos_n, device=G.device, dtype=G.dtype)
    cos_offdiag_mean = float((G - eye).sum() / max(cos_n * (cos_n - 1), 1))

    return {
        "token_norm_mean": token_norm_mean_val,
        "token_norm_std": float(token_norms.std()),
        "token_norm_min": float(token_norms.min()),
        "token_norm_max": float(token_norms.max()),
        "dead_token_fraction": dead_token_fraction,
        "spatial_variance_mean": spatial_variance_mean,
        # Direct collapse indicators
        "mean_token_norm": mean_token_norm,
        "mean_to_token_ratio": mean_to_token_ratio,
        "cos_offdiag_mean": cos_offdiag_mean,
        # Raw (uncentered) spectral health — captures DC collapse
        "sigma_max": spec_raw["sigma_max"],
        "frob": spec_raw["frob"],
        "effective_rank_raw": spec_raw["effective_rank"],
        "stable_rank_raw": spec_raw["stable_rank"],
        # Centred spectral health — captures residual diversity (the
        # historical 'effective_rank' / 'stable_rank' we used to log).
        "effective_rank_centered": spec_centred["effective_rank"],
        "stable_rank_centered": spec_centred["stable_rank"],
        # Aliases for backwards compat with any downstream code still
        # referring to the unqualified names. These point at the centered
        # view, matching the historical behaviour.
        "effective_rank": spec_centred["effective_rank"],
        "stable_rank": spec_centred["stable_rank"],
        "num_tokens": int(B * P_prime),
    }


# ---------------------------------------------------------------------------
# Spectral health of projection weights (W_v-collapse post-mortem from
# K5/T0/K6/T1). Cheap enough to run every 100 steps as Tier A: full SVD of a
# [D, D] matrix at D=256 is ~0.2ms on GPU; at D=1024 ~10ms. We compute it
# lazily on whichever device the weight lives on and cast up to fp32 so the
# singular values don't suffer from bf16 quantisation.
# ---------------------------------------------------------------------------

def weight_spectral_health(W: torch.Tensor) -> Dict[str, float]:
    """Spectral diagnostics for a single weight matrix.

    Args:
        W: ``[d_out, d_in]`` weight tensor. May live on any device / dtype.

    Returns:
        ``{"sigma_max", "sigma_mean", "frob", "effective_rank", "stable_rank"}``
        where:
          * ``sigma_max`` — top singular value. Growing ``sigma_max`` with a
            roughly flat Frobenius norm is the signature of *spectral
            collapse*: all the mass concentrates into a single direction.
          * ``sigma_mean`` — mean singular value. Falls as rank collapses.
          * ``frob`` — :math:`\\sqrt{\\sum_i \\sigma_i^2}`. A stable
            Frobenius norm + rising ``sigma_max`` proves the mass is being
            moved from low-rank tails into the top component, which is the
            W_v-collapse mechanism documented in K5/T0/K6/T1.
          * ``effective_rank`` — :math:`\\exp(-\\sum_i p_i \\log p_i)` with
            :math:`p_i = \\sigma_i^2 / \\sum_j \\sigma_j^2`. Entropy-based
            effective rank. Equals ``min(d_out, d_in)`` for a uniform
            spectrum, 1 for a rank-1 projection. Halving this is the
            collapse signal.
          * ``stable_rank`` — :math:`\\lVert W \\rVert_F^2 / \\sigma_{\\max}^2`.
            Classical stable rank; complements effective_rank.
    """
    W = W.detach().to(torch.float32)
    # SVD of a 2D matrix — prod GPU can handle [1024, 1024] at ~10ms.
    svals = torch.linalg.svdvals(W)
    frob_sq = (svals * svals).sum()
    sigma_max = float(svals.max())
    sigma_mean = float(svals.mean())
    frob = float(frob_sq.sqrt())
    eps = 1e-12
    p = (svals * svals) / frob_sq.clamp_min(eps)
    entropy = -(p * (p.clamp_min(eps).log())).sum()
    effective_rank = float(entropy.exp())
    stable_rank = float(frob_sq / max(sigma_max * sigma_max, eps))
    return {
        "sigma_max": sigma_max,
        "sigma_mean": sigma_mean,
        "frob": frob,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
    }


def probe_decoder_v_proj_spectral(
    mha_modules: Dict[str, nn.Module],
    filter_cross_attn: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Per-layer spectral health of decoder cross-attn V slice.

    Iterates ``find_mha_modules(...)`` keys and extracts the V-side
    spectral diagnostics for every decoder cross-attention MHA
    (``filter_cross_attn`` also keeps the Plan-v2 updater's cross-attn).
    This is the *direct* probe of the "W_v spectral collapse" mechanism
    identified in the K6/T1/K7/T2 post-mortems.

    With the new fused-projection MHA, V no longer lives in a standalone
    ``v_proj`` module. We pull the V-side rows out of the (possibly
    parametrised) joint projection through ``mha.v_weight``:

    * Self-attention: ``v_weight`` is rows ``[2D : 3D]`` of ``qkv_proj``.
    * Cross-attention: ``v_weight`` is rows ``[D : 2D]`` of ``kv_proj``.

    The ``.weight`` accessor on a parametrised ``nn.Linear`` returns the
    runtime weight (post σReparam scaling), so the SVD reflects what the
    forward pass actually used.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, mha in mha_modules.items():
        if filter_cross_attn and "cross_attn" not in name:
            continue
        v_weight = getattr(mha, "v_weight", None)
        if v_weight is None:
            continue
        out[name] = weight_spectral_health(v_weight)
    return out


def probe_updater_spectral(
    model: nn.Module,
) -> Dict[str, Dict[str, float]]:
    """Spectral health of the :class:`PosteriorUpdater`'s q/k/v_proj (I2).

    The decoder cross-attn spectral probe (``probe_decoder_v_proj_spectral``)
    already covers the *decoder* cross-attention — but the posterior
    updater has its own cross-attention with its own W_v, and we have so
    far only measured its Frobenius norm. If the updater's W_v is
    spectrally collapsed, the prior bank receives a low-rank projection
    of CHM and can only specialise into a handful of slots — exactly the
    T4 P4=0.016 symptom.

    Module resolution:
      * Plan v1: no updater → returns ``{}``.
      * Plan v2: ``model.lidar_prior.updater.cross_attn.mha``.
      * Plan v3: ``model.decoder.updater.cross_attn.mha`` (the V3 model
        exposes a ``lidar_prior`` property aliasing ``self.decoder`` so
        the same path ``model.lidar_prior.updater.cross_attn.mha`` works
        for both V2 and V3).

    Returns:
        Dict with keys ``{"updater.cross_attn.mha.q_proj",
        "updater.cross_attn.mha.k_proj", "updater.cross_attn.mha.v_proj"}``
        each mapping to a spectral-health dict. Empty dict if no updater
        exists.
    """
    lidar_prior = getattr(model, "lidar_prior", None)
    if lidar_prior is None:
        return {}
    updater = getattr(lidar_prior, "updater", None)
    if updater is None:
        return {}
    cross_attn = getattr(updater, "cross_attn", None)
    if cross_attn is None:
        return {}
    mha = getattr(cross_attn, "mha", cross_attn)  # CrossAttnSubBlock exposes .mha; plain MHA is itself.

    # Pull q/k/v slices out of the fused projection. With qkv_spectral_norm
    # on, mha.{q,k,v}_weight returns the runtime (parametrised) weight, so
    # the SVD reflects what the forward pass actually used.
    out: Dict[str, Dict[str, float]] = {}
    for slice_name, weight_attr in (
        ("q_proj", "q_weight"),
        ("k_proj", "k_weight"),
        ("v_proj", "v_weight"),
    ):
        weight = getattr(mha, weight_attr, None)
        if weight is None:
            continue
        out[f"updater.cross_attn.mha.{slice_name}"] = weight_spectral_health(weight)
    return out


def probe_slot_usage_histogram(
    decoder_xattn_per_layer: List[torch.Tensor],
    num_prior_tokens: int,
) -> Dict[str, object]:
    """Distribution of decoder cross-attention mass across prior slots (I3).

    For each decoder cross-attn layer, sum post-softmax attention mass
    that lands on each of the first ``M`` prior slots across all queries,
    heads, and batch elements. Normalise to a distribution over M. This
    is complementary to :func:`probe_p4_coverage`:

    * P4 measures "how peaked is each slot's *incoming* attention from
      CHM tokens" — a single-slot measure of specialisation.
    * I3 measures "how evenly does the decoder's *outgoing* attention
      distribute across the M slots" — a slot-bank-wide measure of
      utilisation.

    A bank with P4 coverage 0.016 (T4) could still be healthy if I3
    usage is uniform (all 64 slots contribute equally to decoder
    queries, just not sharply). Conversely an uneven I3 histogram with
    low P4 coverage = a handful of slots carrying nearly all decoder
    attention → the other 60+ slots are dead weight.

    Args:
        decoder_xattn_per_layer: list of ``[B, h, T_q, K]`` attention
            tensors, one per decoder layer. ``K`` is ``M`` for pure-B
            memory or ``M + P'`` when ``concat_chm_to_memory=True``.
        num_prior_tokens: ``M``.

    Returns:
        Dict with:
          * ``usage_per_slot`` — ``[M]`` tensor of normalised slot usage
            (averaged across layers, heads, queries, batch).
          * ``entropy_norm`` — entropy of ``usage_per_slot`` normalised
            by ``log(M)``. 1.0 = perfectly uniform usage, 0.0 = one slot
            gets all mass.
          * ``gini`` — Gini coefficient of ``usage_per_slot``. 0 =
            perfectly uniform, 1 = one slot gets all mass. More robust
            to the long-tail behaviour than entropy for our purposes.
          * ``top1_share`` — max share on any single slot. 1/M = uniform.
          * ``top5_share`` — share going to the 5 most-attended slots.
          * ``active_slot_count`` — number of slots with usage above
            ``1 / (10 M)`` (a soft dead-slot threshold).
          * ``num_slots``.
    """
    per_layer_mass = []
    for attn in decoder_xattn_per_layer:
        attn = attn.detach()
        # Restrict to the prior-slot portion of keys (ignore concat'd CHM keys).
        prior_attn = attn[..., :num_prior_tokens]                   # [B, h, T_q, M]
        # Average across heads and queries; sum across batch.
        layer_usage = prior_attn.mean(dim=(0, 1, 2))                 # [M]
        per_layer_mass.append(layer_usage)

    usage = torch.stack(per_layer_mass, dim=0).mean(dim=0)          # [M]
    # Renormalise to be safe — each row before averaging was a proper
    # distribution over K (first M slots + potentially P' CHM keys),
    # so the sum over M may not be 1.
    usage = usage / usage.sum().clamp(min=1e-12)

    M = int(num_prior_tokens)
    eps = 1e-12
    entropy = float(-(usage * (usage.clamp_min(eps).log())).sum())
    entropy_norm = entropy / max(math.log(max(M, 2)), eps)

    sorted_usage = torch.sort(usage, descending=True).values
    top1_share = float(sorted_usage[0])
    top5_share = float(sorted_usage[: min(5, M)].sum())

    # Gini coefficient: sum over pairs of |uᵢ − uⱼ| / (2·n·mean). Vectorised.
    abs_diffs = (usage.unsqueeze(0) - usage.unsqueeze(1)).abs().mean()
    gini = float(abs_diffs / (2.0 * usage.mean().clamp_min(eps)))

    active_thresh = 1.0 / (10.0 * M)
    active_slot_count = int((usage >= active_thresh).sum())

    return {
        "usage_per_slot": usage.cpu(),                                 # [M]
        "entropy_norm": entropy_norm,
        "gini": gini,
        "top1_share": top1_share,
        "top5_share": top5_share,
        "active_slot_count": active_slot_count,
        "num_slots": M,
    }


# ---------------------------------------------------------------------------
# Attention-concentration probe (cross-attn failure-mode detector).
#
# Cross-attention fails in two complementary ways:
#   1. softmax becomes one-hot (attention entropy → 0, max_attn_weight → 1).
#   2. V-projection mass concentrates into a narrow spectral direction so
#      a single peaked row of attention produces a huge residual.
#
# P1 already captures entropy. This probe adds the pre-softmax logit
# statistics (mean, std, max) and the per-query max post-softmax weight,
# which together pinpoint *when* the attention becomes over-confident and
# whether the logit magnitude is the driver. Together with spectral health
# they completely characterise the K5/T0 failure mode.
# ---------------------------------------------------------------------------

def probe_attention_concentration(
    collector: "AttentionCollector",
) -> Dict[str, Dict[str, float]]:
    """Per-layer pre-softmax logit stats + max post-softmax weight.

    Uses the collector's cached ``(q_source, kv_source, attn_mask)`` so it
    is a few extra matmuls on top of what :func:`probe_p1_entropy` already
    paid. Returns, per MHA::

        {
          "logit_mean", "logit_std", "logit_abs_max",
          "q_norm_mean", "mean_max_attn",
          "layer_kind", "token_kind",
        }

    Interpretation:
        * ``logit_abs_max`` rising with ``logit_std`` → attention is
          becoming a one-hot spike. Combined with low ``effective_rank``
          of v_proj this is the W_v-collapse runaway signature.
        * ``q_norm_mean`` rising alongside ``logit_std`` pinpoints Q-side
          amplification (pre-attention LN γ growth or backbone-token
          magnitude drift).
        * ``mean_max_attn`` > 0.5 for decoder cross-attn indicates the
          decoder has settled into a "one prior slot per query" pattern.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in collector.captured_names:
        entry = collector._cache[name]
        mha = collector._mha_modules[name]
        q_source = entry["q_source"]
        kv_source = entry["kv_source"]
        attn_mask = entry["attn_mask"]

        B, T_q, _ = q_source.shape
        T_kv = kv_source.shape[1]
        h, d_h = mha.n_heads, mha.d_head

        with torch.no_grad():
            Q = mha.project_q(q_source).view(B, T_q, h, d_h).transpose(1, 2)
            K = mha.project_k(kv_source).view(B, T_kv, h, d_h).transpose(1, 2)
            if getattr(mha, "qk_norm", False):
                Q = mha.q_ln(Q)
                K = mha.k_ln(K)
            logits = (Q @ K.transpose(-2, -1)) / mha.scale        # [B, h, T_q, T_kv]

            if attn_mask is not None:
                masked = logits.masked_fill(attn_mask, float("nan"))
                valid = masked.reshape(-1)
                valid = valid[~torch.isnan(valid)]
                attn_for_soft = logits.masked_fill(attn_mask, float("-inf"))
            else:
                valid = logits.reshape(-1)
                attn_for_soft = logits

            attn = attn_for_soft.softmax(dim=-1)                   # [B, h, T_q, T_kv]
            max_per_q = attn.max(dim=-1).values                    # [B, h, T_q]
            q_norm = Q.norm(dim=-1).reshape(-1)                    # per-token Q norm

        out[name] = {
            "layer_kind": _layer_kind(name),
            "token_kind": _token_kind(name),
            "logit_mean": float(valid.mean()),
            "logit_std": float(valid.std()),
            "logit_abs_max": float(valid.abs().max()),
            "q_norm_mean": float(q_norm.mean()),
            "q_norm_std": float(q_norm.std()),
            "mean_max_attn": float(max_per_q.mean()),
            "max_max_attn": float(max_per_q.max()),
            "T_kv": int(T_kv),
        }
    return out


# ---------------------------------------------------------------------------
# Health flags — derived booleans from the probe dict
# ---------------------------------------------------------------------------

def compute_health_flags(
    p1_summary: Optional[Dict[str, float]] = None,
    p3: Optional[Dict[str, float]] = None,
    p4: Optional[Dict[str, float]] = None,
    prior_bank_health: Optional[Dict[str, float]] = None,
    spectral: Optional[Dict[str, Dict[str, float]]] = None,
    concentration: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, bool]:
    """Derive health booleans from probe summaries.

    Returns a dict of ``<flag>: bool``. Missing inputs (e.g. Plan v1 has no
    P3/P4) make the flags default to ``False`` — we never raise a risk we
    cannot verify.
    """
    th = THRESHOLDS

    mean_delta = p3.get("mean_delta") if p3 else None
    coverage_ratio = p4.get("coverage_ratio") if p4 else None
    updater_H = p1_summary.get("updater_xattn_mean_H") if p1_summary else None
    dead_slot_fraction = prior_bank_health.get("dead_slot_fraction") if prior_bank_health else None

    prior_collapse = (
        mean_delta is not None
        and mean_delta > th["prior_collapse_mean_delta"]
    )
    image_agnostic = (
        mean_delta is not None
        and mean_delta < th["image_agnostic_mean_delta"]
    )
    slot_collapse = (
        coverage_ratio is not None
        and updater_H is not None
        and not math.isnan(updater_H)
        and coverage_ratio < th["slot_collapse_coverage_ratio"]
        and updater_H < th["slot_collapse_updater_H"]
    )
    dead_slots_many = (
        dead_slot_fraction is not None
        and dead_slot_fraction > 0.25
    )

    # ---- Decoder cross-attn spectral collapse (W_v trap) ---------------
    # Use min across decoder cross-attn layers; one sick layer corrupts the
    # whole chain. Updater cross-attn is excluded (different dynamics).
    spectral_collapse_risk = False
    if spectral:
        decoder_sr = [
            v["stable_rank"] for n, v in spectral.items()
            if n.startswith("decoder.layers.") and "cross_attn" in n
        ]
        if decoder_sr and min(decoder_sr) < th["spectral_collapse_stable_rank"]:
            spectral_collapse_risk = True
    one_hot_attn_risk = False
    if concentration:
        decoder_mx = [
            v["mean_max_attn"] for n, v in concentration.items()
            if n.startswith("decoder.layers.") and "cross_attn" in n
        ]
        if decoder_mx and max(decoder_mx) > th["spectral_collapse_max_attn"]:
            one_hot_attn_risk = True

    return {
        "prior_collapse_risk": bool(prior_collapse),
        "image_agnostic_risk": bool(image_agnostic),
        "slot_collapse_risk": bool(slot_collapse),
        "dead_slots_excessive": bool(dead_slots_many),
        "spectral_collapse_risk": bool(spectral_collapse_risk),
        "one_hot_attn_risk": bool(one_hot_attn_risk),
    }


# ---------------------------------------------------------------------------
# Reliance map → RGB tensor for TB
# ---------------------------------------------------------------------------

def reliance_map_to_rgb(reliance_map: torch.Tensor) -> torch.Tensor:
    """Convert a ``[B, h, w]`` reliance map to a ``[3, h, (B*w)]`` RGB grid.

    Uses a simple Red/Green colormap — red where the decoder leans on the
    prior, green where it leans on raw CHM evidence. Works without matplotlib
    so the callback has no hard extra dependency.
    """
    r_mask = reliance_map.clamp(0, 1)
    g_mask = 1.0 - r_mask
    b_mask = torch.zeros_like(r_mask)
    rgb = torch.stack([r_mask, g_mask, b_mask], dim=0)              # [3, B, h, w]
    return rgb.transpose(1, 2).reshape(3, rgb.shape[2], -1)          # [3, h, B*w]


# ---------------------------------------------------------------------------
# Heatmap → RGB tensor for TB
# ---------------------------------------------------------------------------

def heatmap_to_rgb(heatmap: torch.Tensor) -> torch.Tensor:
    """Convert a ``[H, W]`` non-negative heatmap to a ``[3, H, W]`` RGB tensor.

    Simple viridis-free mapping: normalize to [0, 1] then use the R channel.
    Keeps the function pure-torch; matplotlib is optional in the callback.
    """
    h = heatmap.float()
    lo, hi = h.min(), h.max()
    if (hi - lo) < 1e-8:
        h = torch.zeros_like(h)
    else:
        h = (h - lo) / (hi - lo)
    return torch.stack([h, h.pow(0.5), h.pow(2.0)], dim=0)
