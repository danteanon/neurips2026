"""Lightning callback that streams CHM diagnostics (P1-P5) to TensorBoard.

Attach to any trainer whose Lightning module holds a height estimation model
(Plan v1 single / Plan v1 DPT / Plan v2 single / Plan v2 DPT / Plan v3 DPT).
The callback auto-detects the architecture and skips Plan-v2-only probes for
Plan v1 runs.

Three logging timelines, each against a different x-axis on TensorBoard:

* ``diag/``       — per validation epoch (x=epoch). The canonical P1-P5 probe
  suite on a 4-sample cached val batch, in ``eval()`` mode.
* ``diag_train/`` — per N training steps (x=global_step). Two tiers:
    - *Tier A* (every ``train_log_tier_a_every_n_steps``): cheap scalar reads
      of the updater's gate parameter (α / tanh(α) / γ / sigmoid(z)), the
      cross-attention weight norms (``q_proj``, ``k_proj``, ``v_proj``,
      ``out_proj``), pre-attention LayerNorm γ norms, the prior bank mean
      norm, and **spectral diagnostics** on every decoder cross-attn
      v_proj (σ_max / effective rank / Frobenius / stable_rank — direct
      probe of the W_v-collapse mechanism documented in the K5/T0/K6/T1
      post-mortems). These fire fast-moving signals long before the val
      epoch would catch them.
    - *Tier B* (every ``train_log_tier_b_every_n_steps``): full P1/P2/P3
      probe suite on a cached 4-sample *training* batch in ``eval()`` mode
      **plus** per-layer pre-softmax logit stats + mean-max attention
      weight (the attention-concentration probe — complements the spectral
      probe by isolating *softmax-spike* failures from *W_v collapse*).

Consult :mod:`utils.height_diagnostics` for the probe definitions. See
``docs/chm/ablation_plan.md`` section 7.1 for the canonical list of TB keys
and how to interpret them while a run is in flight.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from lightning.pytorch.callbacks import Callback

from utils.height_diagnostics import (
    AttentionCollector,
    compute_health_flags,
    find_mha_modules,
    heatmap_to_rgb,
    probe_attention_concentration,
    probe_chm_encoder_stats,
    probe_decoder_v_proj_spectral,
    probe_p1_entropy,
    probe_p2_norms,
    probe_p3_delta,
    probe_p4_coverage,
    probe_p5_reliance,
    probe_prior_bank_health,
    probe_slot_usage_histogram,
    probe_updater_spectral,
    reliance_map_to_rgb,
    summarise_p1,
    summarise_p2,
    weight_spectral_health,
)

logger = logging.getLogger(__name__)


def _normalise_batch(batch):
    """Flatten ``per_source_collate_split`` batches into a single tuple.

    Lightning hands callbacks the *raw* batch from the dataloader, before
    ``HeightEstimationModule.{training,validation}_step`` does any per-source
    routing. With ``collate_fn: per_source_collate`` the batch is a flat
    tuple ``(imgs, chm, gt, ...)``; with ``collate_fn: per_source_collate_split``
    it is a list of such tuples — one per source — and any code that does
    ``batch[2]`` will index INTO sub-batch 2 (which doesn't exist for a
    2-source mix) instead of the third tensor field.

    Detection: the batch is a list/tuple whose first element is itself a
    list/tuple of tensors → it's the split form. We concatenate each tensor
    field across sources so downstream diagnostics see a single representative
    batch (works because the model is source-agnostic at forward time and
    the callbacks only care about ``imgs``, ``chm``, ``gt`` shapes).
    """
    if (
        isinstance(batch, (list, tuple))
        and len(batch) > 0
        and isinstance(batch[0], (list, tuple))
    ):
        # split-collation form: concatenate each field across sub-batches.
        # Use the smallest field-arity in case sources emit different
        # tuple lengths (e.g. one with contrastive views, one without).
        n_fields = min(len(sub) for sub in batch)
        return tuple(
            torch.cat([sub[i] for sub in batch], dim=0)
            for i in range(n_fields)
        )
    return batch


class CHMDiagnosticsCallback(Callback):
    """Per-epoch Track A diagnostics shipped to TensorBoard.

    What it does, every time Lightning completes a validation epoch:

    1. Uses a cached 4-sample batch (captured once on the first val batch of
       epoch 0) so the numbers are comparable across epochs.
    2. Runs the model in ``eval()`` mode with ``return_intermediates=True``
       inside an :class:`AttentionCollector` context — this captures the
       MHA inputs and exposes recomputable attention weights.
    3. Computes probes P1 (entropy), P2 (V-norms), and — if the model has a
       ``lidar_prior`` — P3 (posterior-prior delta) and P4 (slot coverage).
       P5 (prior-vs-evidence reliance) runs only when
       ``concat_chm_to_memory=True``.
    4. Derives health flags from the probe summaries.
    5. Logs everything to the trainer's TB logger under ``diag/`` (scalars),
       ``diag_hist/`` (histograms) and ``diag_img/`` (images).

    Disables itself gracefully (with an ``info`` log line) when:
      - the Lightning module doesn't expose a height-style model (no
        ``decoder.layers`` attribute),
      - the logger isn't TensorBoard-compatible (histograms/images are
        silently skipped, scalars still go through ``pl_module.log``).

    Cost: one extra forward pass per val epoch on the cached val batch
    (size = ``val_batch_size`` by default; see ``diagnostic_batch_size``).
    With ``val_batch_size=8`` and a ViT-L backbone this is about 1–2
    seconds per epoch — negligible vs the rest of validation.

    Diagnostic-batch sizing
    -----------------------
    The variance / covariance / cosine statistics in ``probe_chm_encoder_stats``
    are estimated over the batch axis, so the sample size matters. With
    ``N=4`` (the previous default), the per-channel variance estimate has
    ~40% relative noise and the ``unbiased=False`` reduction is biased
    low by ``(N-1)/N = 25%``. This led to ``spatial_variance_mean`` reading
    one specific 4-image batch as ~50× lower than the encoder's actual
    behaviour on representative val data. The new default — ``None``,
    meaning "use the full val batch" — sidesteps both issues without
    needing to know ``val_batch_size`` ahead of time.

    Set ``diagnostic_batch_size`` to a positive int to override (e.g. for
    tiny-VRAM debug runs where the full val batch would OOM the extra
    forward).
    """

    def __init__(
        self,
        enabled: bool = True,
        diagnostic_batch_size: Optional[int] = None,
        log_histograms: bool = True,
        log_images: bool = True,
        image_log_every_n_epochs: int = 5,
        enable_train_logging: bool = True,
        train_log_tier_a_every_n_steps: int = 100,
        train_log_tier_b_every_n_steps: int = 500,
    ):
        super().__init__()
        self.enabled = enabled
        # ``None`` => use the full val batch (whatever val_batch_size is).
        # Positive int => cap to that size. The non-None path is kept as an
        # escape hatch for memory-constrained setups.
        self.diagnostic_batch_size = (
            None if diagnostic_batch_size is None
            else max(1, int(diagnostic_batch_size))
        )
        self.log_histograms = log_histograms
        self.log_images = log_images
        self.image_log_every_n_epochs = max(1, int(image_log_every_n_epochs))

        # Train-time (tier A / tier B) logging knobs. Tier A is a cheap
        # parameter-read pass (gate scalars + weight norms) that fires every
        # N steps. Tier B runs P1-P3 probes on a cached train batch and is
        # more expensive, so fires less often.
        self.enable_train_logging = bool(enable_train_logging)
        self.train_log_tier_a_every_n_steps = max(1, int(train_log_tier_a_every_n_steps))
        self.train_log_tier_b_every_n_steps = max(1, int(train_log_tier_b_every_n_steps))

        self._diag_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._train_probe_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._mha_modules: Optional[dict] = None
        self._inspected_once = False
        self._disabled_reason: Optional[str] = None
        self._warned_once = False
        self._pl_module_ref = None

    # ------------------------------------------------------------------
    # Inspection + batch capture
    # ------------------------------------------------------------------
    def _inspect_model(self, pl_module) -> None:
        """One-time introspection: find MHA modules + record architecture flags."""
        self._inspected_once = True
        if not self.enabled:
            self._disabled_reason = "explicitly disabled via config"
            return

        model = getattr(pl_module, "model", None)
        if model is None:
            self._disabled_reason = "pl_module has no `.model` attribute"
            return

        mhas = find_mha_modules(model)
        if not mhas:
            self._disabled_reason = "no MHA modules found (not a height-style model)"
            return

        self._mha_modules = mhas
        logger.info(
            f"CHMDiagnosticsCallback active — tracking {len(mhas)} MHA modules "
            f"(has_lidar_prior={getattr(model, 'lidar_prior', None) is not None})"
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not self._inspected_once:
            self._inspect_model(pl_module)
        if self._disabled_reason is not None:
            return
        if self._diag_batch is not None:
            return
        if batch_idx != 0:
            return

        # Capture the first batch we see; use a detached clone so subsequent
        # val steps don't mutate it. Stays on GPU so the re-forward is fast.
        # By default keep the FULL val batch (matches ``val_batch_size``) so
        # variance / covariance estimates have the same sample size as the
        # actual training-time loss computations. Subset only if
        # ``diagnostic_batch_size`` was explicitly set.
        batch = _normalise_batch(batch)
        imgs, chm, gt = batch[0], batch[1], batch[2]
        if self.diagnostic_batch_size is None:
            n = imgs.size(0)
        else:
            n = min(self.diagnostic_batch_size, imgs.size(0))
        self._diag_batch = (
            imgs[:n].detach().clone(),
            chm[:n].detach().clone(),
            gt[:n].detach().clone(),
        )

    # ------------------------------------------------------------------
    # Main probing entry point
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        if self._disabled_reason is not None:
            if self._disabled_reason and not getattr(self, "_warned_once", False):
                logger.info(f"CHMDiagnosticsCallback disabled: {self._disabled_reason}")
                self._warned_once = True
            return
        if self._diag_batch is None:
            return
        if trainer.sanity_checking:
            return

        if not trainer.is_global_zero:
            return

        try:
            self._run_probes(trainer, pl_module)
        except Exception as exc:  # diagnostics must never crash training
            logger.warning(
                f"CHMDiagnosticsCallback failed on epoch {trainer.current_epoch}: "
                f"{type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Train-time hooks: Tier A (cheap param reads) + Tier B (cached-batch
    # P1-P3 re-forward). Both live under the ``diag_train/`` TB namespace
    # and use ``trainer.global_step`` as the x-axis so they plot on the same
    # timeline as the loss curves.
    # ------------------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.enable_train_logging:
            return
        if not self._inspected_once:
            self._inspect_model(pl_module)
        if self._disabled_reason is not None:
            return
        if self._train_probe_batch is not None:
            return
        # Capture the very first training batch we see as the tier-B probe
        # fixture. Clone+detach so subsequent train steps don't mutate it.
        # Stays on GPU so the tier-B re-forward is fast.
        try:
            batch = _normalise_batch(batch)
            imgs, chm, gt = batch[0], batch[1], batch[2]
        except (TypeError, ValueError, IndexError):
            return
        if self.diagnostic_batch_size is None:
            n = imgs.size(0)
        else:
            n = min(self.diagnostic_batch_size, imgs.size(0))
        self._train_probe_batch = (
            imgs[:n].detach().clone(),
            chm[:n].detach().clone(),
            gt[:n].detach().clone(),
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if not self.enable_train_logging:
            return
        if self._disabled_reason is not None:
            return
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return

        step = int(trainer.global_step)

        # Tier A — cheap parameter reads. Fires from step 0 (captures init).
        if step % self.train_log_tier_a_every_n_steps == 0:
            try:
                self._log_train_tier_a(pl_module, step)
            except Exception as exc:
                logger.warning(
                    f"CHMDiagnosticsCallback tier-A (updater) failed at "
                    f"step {step}: {type(exc).__name__}: {exc}"
                )
            try:
                self._log_train_tier_a_decoder(pl_module, step)
            except Exception as exc:
                logger.warning(
                    f"CHMDiagnosticsCallback tier-A (decoder spectral) "
                    f"failed at step {step}: {type(exc).__name__}: {exc}"
                )
            try:
                self._log_train_tier_a_useries(pl_module, step)
            except Exception as exc:
                logger.warning(
                    f"CHMDiagnosticsCallback tier-A (U-series) "
                    f"failed at step {step}: {type(exc).__name__}: {exc}"
                )

        # Tier B — cached-batch probe re-forward. Needs a probe batch first.
        if (
            step > 0
            and step % self.train_log_tier_b_every_n_steps == 0
            and self._train_probe_batch is not None
        ):
            try:
                self._run_train_tier_b(trainer, pl_module, step)
            except Exception as exc:
                logger.warning(
                    f"CHMDiagnosticsCallback tier-B failed at step {step}: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ------------------------------------------------------------------
    # Tier A — scalar reads of updater parameters. No forward pass, no
    # cached batch required. Fires early enough that the "W_v magnitude
    # trap" (K5/T0 post-mortem) shows up in real time, not once per epoch.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _log_train_tier_a(self, pl_module, step: int) -> None:
        model = getattr(pl_module, "model", None)
        if model is None:
            return
        lidar_prior = getattr(model, "lidar_prior", None)
        if lidar_prior is None:
            return  # Plan v1 — no updater to probe.
        updater = getattr(lidar_prior, "updater", None)
        if updater is None:
            return

        # --- Gate-specific scalar(s): raw + effective residual strength ---
        gate_type = getattr(updater, "gate_type", None)
        if gate_type == "post_ln_alpha":
            alpha = updater.alpha.detach()
            self._log_scalar("diag_train/gate/alpha", float(alpha), step)
            self._log_scalar(
                "diag_train/gate/effective", float(alpha.abs()), step
            )
            if hasattr(updater, "norm_update"):
                ln_w = updater.norm_update.weight.detach()
                self._log_scalar(
                    "diag_train/updater/ln_update_gamma_mean",
                    float(ln_w.mean()),
                    step,
                )
                self._log_scalar(
                    "diag_train/updater/ln_update_gamma_norm",
                    float(ln_w.norm()),
                    step,
                )
        elif gate_type == "tanh":
            alpha = updater.alpha.detach()
            self._log_scalar("diag_train/gate/alpha", float(alpha), step)
            self._log_scalar(
                "diag_train/gate/effective", float(torch.tanh(alpha)), step
            )
        elif gate_type == "alpha":
            # U-bank Lever 2: posterior = prior + α · update (no LN, no
            # tanh). Tracking ``alpha`` directly is the headline number —
            # if α saturates near 1.0 the bank is being rewritten every
            # iteration; if α stays near init (~0.03) the bank is being
            # gently refined.
            alpha = updater.alpha.detach()
            self._log_scalar("diag_train/gate/alpha", float(alpha), step)
            self._log_scalar(
                "diag_train/gate/effective", float(alpha.abs()), step
            )
        elif gate_type == "layerscale":
            gamma = updater.gamma.detach()
            self._log_scalar("diag_train/gate/gamma_mean", float(gamma.mean()), step)
            self._log_scalar("diag_train/gate/gamma_std", float(gamma.std()), step)
            self._log_scalar("diag_train/gate/gamma_abs_max", float(gamma.abs().max()), step)
            self._log_scalar("diag_train/gate/effective", float(gamma.norm()), step)
            # Histogram is cheap for a single [d] vector; aids rank analysis.
            self._log_histogram("diag_train_hist/gate/gamma", gamma.cpu(), step)
        elif gate_type == "gru":
            cell = updater.gru
            H = cell.hidden_size
            # Effective update-gate bias = b_iz + b_hz (slice [H:2H] from each).
            z_bias = (
                cell.bias_ih[H : 2 * H].detach() + cell.bias_hh[H : 2 * H].detach()
            )
            self._log_scalar("diag_train/gate/z_bias_mean", float(z_bias.mean()), step)
            self._log_scalar(
                "diag_train/gate/z_gate_mean",
                float(torch.sigmoid(z_bias).mean()),
                step,
            )
            # "Effective" residual strength = 1 - sigmoid(z_bias) ≈ how much of
            # the candidate n leaks into the posterior (near-no-op ⇒ 0).
            self._log_scalar(
                "diag_train/gate/effective",
                float((1.0 - torch.sigmoid(z_bias)).mean()),
                step,
            )

        # --- Cross-attention projection weight norms. These are THE signal
        # for the W_v magnitude trap: v_proj norm growing unchecked is the
        # root cause of the K5/T0 divergence. q_proj/k_proj/out_proj are
        # logged for context (they move together in a balanced MHA).
        #
        # The new fused MHA stores Q in ``q_proj.weight`` and K/V as the
        # two halves of ``kv_proj.weight`` (cross-attn layout). We expose
        # them via the ``{q,k,v}_weight`` properties so the TB tag names
        # remain stable even though there is no longer a literal ``v_proj``
        # module. ``out_proj`` is unchanged.
        xattn = getattr(updater, "cross_attn", None)
        if xattn is not None:
            for proj_name, weight_attr in (
                ("q_proj", "q_weight"),
                ("k_proj", "k_weight"),
                ("v_proj", "v_weight"),
            ):
                weight = getattr(xattn, weight_attr, None)
                if weight is not None:
                    self._log_scalar(
                        f"diag_train/updater/{proj_name}_weight_norm",
                        float(weight.detach().norm()),
                        step,
                    )
            out_proj = getattr(xattn, "out_proj", None)
            if out_proj is not None and hasattr(out_proj, "weight"):
                self._log_scalar(
                    "diag_train/updater/out_proj_weight_norm",
                    float(out_proj.weight.detach().norm()),
                    step,
                )

        # --- Pre-attention LayerNorm gammas. Growing norms here indicate
        # LN learning to amplify its input rather than normalise it — another
        # way the updater can end up multiplying its residual without α ever
        # moving.
        for ln_name in ("norm_q", "norm_kv"):
            ln = getattr(updater, ln_name, None)
            if ln is not None and hasattr(ln, "weight"):
                self._log_scalar(
                    f"diag_train/updater/{ln_name}_gamma_norm",
                    float(ln.weight.detach().norm()),
                    step,
                )

        # --- Prior bank health. Cheap and extremely informative for slot
        # collapse: a shrinking mean norm means the prior is being pulled
        # toward zero by the residual add.
        prior = getattr(lidar_prior, "prior", None)
        if prior is not None:
            prior_det = prior.detach()
            per_slot_norm = prior_det.norm(dim=-1)
            self._log_scalar(
                "diag_train/prior_bank/mean_norm", float(per_slot_norm.mean()), step
            )
            self._log_scalar(
                "diag_train/prior_bank/std_norm", float(per_slot_norm.std()), step
            )

        # --- Slot identity embedding (DETR query_embed pattern). When
        # absent (default) all of these are skipped so legacy runs are
        # bit-identical. The headline number is `q_input_offdiag_cosine`:
        # mean off-diagonal cosine similarity of (prior + slot_embed),
        # i.e. the actual input to `norm_q`. Lower magnitude ⇒ slots ask
        # more orthogonal questions ⇒ updater attention can specialise
        # per slot. At init the value is ~0 (random vectors in 1024-D
        # are nearly orthogonal); it should stay small or drop further
        # if slot_embed is doing its job.
        slot_embed = getattr(lidar_prior, "slot_embed", None)
        if slot_embed is not None:
            slot_det = slot_embed.detach()
            per_slot_norm = slot_det.norm(dim=-1)
            self._log_scalar(
                "diag_train/slot_embed/mean_norm",
                float(per_slot_norm.mean()),
                step,
            )
            self._log_scalar(
                "diag_train/slot_embed/std_norm",
                float(per_slot_norm.std()),
                step,
            )
            if prior is not None and prior.shape == slot_embed.shape:
                q_input = prior.detach() + slot_det
                norms = q_input.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                normed = q_input / norms
                cos = normed @ normed.T
                M = cos.shape[0]
                if M > 1:
                    off_mask = ~torch.eye(M, dtype=torch.bool, device=cos.device)
                    off = cos[off_mask]
                    self._log_scalar(
                        "diag_train/slot_embed/q_input_offdiag_cosine_mean",
                        float(off.mean()),
                        step,
                    )
                    self._log_scalar(
                        "diag_train/slot_embed/q_input_offdiag_cosine_abs_max",
                        float(off.abs().max()),
                        step,
                    )

    @torch.no_grad()
    def _log_train_tier_a_decoder(self, pl_module, step: int) -> None:
        """Decoder cross-attn Tier A: per-layer weight norms + spectral health.

        This is the failure-point probe identified in the K5/T0/K6/T1
        post-mortems: decoder cross-attn v_proj undergoes *spectral
        collapse* (σ_max triples, effective rank halves, stable rank → 1)
        while the Frobenius norm stays flat. Log per-layer so we can
        pinpoint which decoder layer snaps first and decide whether the
        problem is monotonic across depth or localised.
        """
        if self._mha_modules is None:
            return

        model = getattr(pl_module, "model", None)
        if model is None:
            return

        # (1) Per-layer Frobenius norms for q/k/v/out — direct W_v magnitude
        #     trap detector. Cheap (4 `.norm()` calls per layer).
        #
        # The new fused MHA layout means there is no longer a literal
        # ``v_proj`` module on the decoder cross-attn (V is the second half
        # of the joint ``kv_proj`` weight). We pull the runtime Q/K/V slices
        # out of the (possibly parametrised) fused projection through the
        # ``{q,k,v}_weight`` properties so the TB tag names remain stable.
        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "layers"):
            for i, layer in enumerate(decoder.layers):
                xa = getattr(layer, "cross_attn", None)
                if xa is None or not hasattr(xa, "mha"):
                    continue
                mha = xa.mha
                for proj_name, weight_attr in (
                    ("q_proj", "q_weight"),
                    ("k_proj", "k_weight"),
                    ("v_proj", "v_weight"),
                ):
                    weight = getattr(mha, weight_attr, None)
                    if weight is None:
                        continue
                    self._log_scalar(
                        f"diag_train/decoder/L{i}/{proj_name}_frob",
                        float(weight.detach().norm()),
                        step,
                    )
                out_proj = getattr(mha, "out_proj", None)
                if out_proj is not None and hasattr(out_proj, "weight"):
                    self._log_scalar(
                        f"diag_train/decoder/L{i}/out_proj_frob",
                        float(out_proj.weight.detach().norm()),
                        step,
                    )

        # (2) Per-layer spectral diagnostics (SVD-based) for v_proj only.
        #     This is *the* canonical signal for the spectral-collapse
        #     mechanism: σ_max rising, eff_rank falling, stable_rank → 1.
        try:
            spectral = probe_decoder_v_proj_spectral(
                self._mha_modules, filter_cross_attn=True
            )
        except Exception as exc:
            logger.debug(f"skip spectral probe at step {step}: {exc}")
            return

        for name, stats in spectral.items():
            # Strip the trailing ``.cross_attn.mha`` / ``.cross_attn`` so the
            # TB tag reads ``diag_train/spectral/decoder.layers.0/sigma_max``.
            short = name.replace(".cross_attn.mha", "").replace(".cross_attn", "")
            self._log_scalar(f"diag_train/spectral/{short}/sigma_max", stats["sigma_max"], step)
            self._log_scalar(f"diag_train/spectral/{short}/sigma_mean", stats["sigma_mean"], step)
            self._log_scalar(f"diag_train/spectral/{short}/frob", stats["frob"], step)
            self._log_scalar(
                f"diag_train/spectral/{short}/effective_rank",
                stats["effective_rank"],
                step,
            )
            self._log_scalar(
                f"diag_train/spectral/{short}/stable_rank",
                stats["stable_rank"],
                step,
            )

        # (3) Aggregate min / max across decoder cross-attn layers. Makes
        #     early-warning thresholds trivial to set in TB.
        decoder_stats = [v for n, v in spectral.items() if n.startswith("decoder.layers.")]
        if decoder_stats:
            self._log_scalar(
                "diag_train/spectral/decoder_summary/max_sigma_max",
                max(v["sigma_max"] for v in decoder_stats),
                step,
            )
            self._log_scalar(
                "diag_train/spectral/decoder_summary/min_effective_rank",
                min(v["effective_rank"] for v in decoder_stats),
                step,
            )
            self._log_scalar(
                "diag_train/spectral/decoder_summary/min_stable_rank",
                min(v["stable_rank"] for v in decoder_stats),
                step,
            )

        # (4) Updater cross-attn q/k/v spectral health (I2). The updater's
        #     W_v spectral collapse is the upstream sibling of the decoder
        #     W_v collapse: if it's low-rank, the prior bank receives a
        #     low-rank projection of CHM and only a handful of slots can
        #     specialise (T4 P4=0.016 symptom).
        try:
            updater_spec = probe_updater_spectral(model)
        except Exception as exc:
            logger.debug(f"skip updater-spectral probe at step {step}: {exc}")
            updater_spec = {}
        for name, stats in updater_spec.items():
            for key in ("sigma_max", "sigma_mean", "frob", "effective_rank", "stable_rank"):
                self._log_scalar(f"diag_train/spectral/{name}/{key}", stats[key], step)

    # ------------------------------------------------------------------
    # Tier A — U-series-specific probes: σReparam γ trajectory + rank
    # regulariser stable-rank summary + activation regulariser RMS readout.
    # All gracefully no-op when the corresponding feature is off.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _log_train_tier_a_useries(self, pl_module, step: int) -> None:
        """Log U-series feature health.

        Three blocks (each conditional on the feature being enabled):

        * **σReparam γ statistics** — every linear that has had a
          ``_SigmaReparamScale`` parametrisation registered exposes a
          ``parametrizations.weight[0].gamma`` ``nn.Parameter``. We
          aggregate them into a single histogram + mean / min / max
          scalars under ``diag_train/sreparam/gamma_*``. The σReparam
          paper's diagnostic lever: γ should grow toward σ_max(W_init)
          (~3-5 for our decoder) over training; if it stays pinned at
          init the regulariser is functionally inactive — that's the
          T11 frozen-γ symptom.
        * **Rank-floor regulariser summary** — when ``rank_reg`` is
          attached to the Lightning module, we re-compute stable_rank
          per-target-linear, take min / mean / max, and log them as
          ``diag_train/u_rank/{min,mean,max}_stable_rank``.
        * **Activation regulariser readout** — when ``act_reg`` is
          attached, the most recent training step's per-site mean RMS
          values are surfaced as ``diag_train/u_act/site_*/rms``.
        """
        model = getattr(pl_module, "model", None)
        if model is None:
            return

        # --- σReparam γ statistics across every parametrised linear ---
        gamma_values = []
        for module_name, module in model.named_modules():
            paramz = getattr(module, "parametrizations", None)
            if paramz is None or "weight" not in paramz:
                continue
            for entry in paramz["weight"]:
                gamma = getattr(entry, "gamma", None)
                if gamma is not None and gamma.ndim == 0:
                    gamma_values.append((module_name, float(gamma.detach())))
        if gamma_values:
            vals = torch.tensor([v for _, v in gamma_values])
            self._log_scalar("diag_train/sreparam/gamma_mean",
                             float(vals.mean()), step)
            self._log_scalar("diag_train/sreparam/gamma_min",
                             float(vals.min()), step)
            self._log_scalar("diag_train/sreparam/gamma_max",
                             float(vals.max()), step)
            self._log_scalar("diag_train/sreparam/gamma_std",
                             float(vals.std() if len(vals) > 1 else 0.0), step)
            self._log_histogram("diag_train_hist/sreparam/gamma", vals, step)

        # --- Rank-floor regulariser summary -------------------------
        rank_reg = getattr(pl_module, "rank_reg", None)
        if rank_reg is not None and rank_reg.num_target_linears > 0:
            try:
                summary = rank_reg.stable_rank_summary()
            except Exception as exc:
                logger.debug(f"rank_reg summary failed at step {step}: {exc}")
                summary = {}
            if summary:
                vals = torch.tensor(list(summary.values()))
                self._log_scalar("diag_train/u_rank/min_stable_rank",
                                 float(vals.min()), step)
                self._log_scalar("diag_train/u_rank/mean_stable_rank",
                                 float(vals.mean()), step)
                self._log_scalar("diag_train/u_rank/max_stable_rank",
                                 float(vals.max()), step)
                self._log_histogram(
                    "diag_train_hist/u_rank/stable_rank", vals, step
                )

        # --- Activation regulariser readout -------------------------
        act_reg = getattr(pl_module, "act_reg", None)
        if act_reg is not None:
            try:
                rms_list = act_reg.last_norms()
            except Exception as exc:
                logger.debug(f"act_reg last_norms failed at step {step}: {exc}")
                rms_list = []
            if rms_list:
                vals = torch.tensor(rms_list)
                self._log_scalar("diag_train/u_act/rms_mean",
                                 float(vals.mean()), step)
                self._log_scalar("diag_train/u_act/rms_min",
                                 float(vals.min()), step)
                self._log_scalar("diag_train/u_act/rms_max",
                                 float(vals.max()), step)

    # ------------------------------------------------------------------
    # Tier B — full P1/P2/P3 on the cached train batch. Same code path as
    # the per-val-epoch probes; just runs on a train-side cached batch and
    # logs under the ``diag_train/`` namespace.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_train_tier_b(self, trainer, pl_module, step: int) -> None:
        assert self._train_probe_batch is not None
        if self._mha_modules is None:
            # Inspection hasn't happened (or model isn't a height model).
            return

        model = pl_module.model
        device = pl_module.device
        was_training = model.training
        model.eval()

        imgs, chm, _gt = self._train_probe_batch
        imgs = imgs.to(device, non_blocking=True)
        chm = chm.to(device, non_blocking=True)

        has_prior = getattr(model, "lidar_prior", None) is not None
        concat_chm_to_memory = bool(
            getattr(model, "concat_chm_to_memory", False)
        ) if has_prior else False

        try:
            with AttentionCollector(self._mha_modules) as collector:
                out = model(imgs, chm, return_intermediates=True)
                if isinstance(out, tuple) and len(out) >= 2:
                    _pred, intermediates = out[0], out[1]
                else:  # pragma: no cover — defensive
                    raise RuntimeError(
                        "Expected model(..., return_intermediates=True) to return a tuple."
                    )

                p1 = probe_p1_entropy(collector)
                p2 = probe_p2_norms(collector)
                p1_summary = summarise_p1(p1)
                p2_summary = summarise_p2(p2)

                self._log_p1(p1, p1_summary, step, prefix="diag_train")
                self._log_p2(p2, p2_summary, step, prefix="diag_train")

                # Attention-concentration probe — per-layer pre-softmax
                # logit stats + max attention weight. Pinpoints the
                # one-hot-spike failure mode complementary to W_v collapse.
                concentration = probe_attention_concentration(collector)
                self._log_concentration(concentration, step, prefix="diag_train")

                # --- CHM encoder stats (I1) -------------------------------
                # V1: raw encoder tokens live in ``chm_memory`` (chm_tokens+PE).
                # V2/V3: raw encoder tokens live in ``chm_tokens_with_pe``
                #        (``chm_memory`` is the posterior). Prefer the raw
                #        tokens when both exist so the probe reports CHM
                #        encoder health, not posterior health.
                chm_enc_source = intermediates.get("chm_tokens_with_pe")
                if chm_enc_source is None and not has_prior:
                    chm_enc_source = intermediates.get("chm_memory")
                if chm_enc_source is not None:
                    try:
                        chm_stats = probe_chm_encoder_stats(chm_enc_source)
                        self._log_chm_encoder(chm_stats, step, prefix="diag_train")
                    except Exception as exc:
                        logger.debug(f"skip CHM encoder probe at step {step}: {exc}")

                p3 = None
                bank_health = None
                if has_prior:
                    prior_bank = intermediates.get("prior_bank")
                    chm_memory = intermediates.get("chm_memory")
                    num_prior_tokens = int(model.lidar_prior.num_prior_tokens)

                    if prior_bank is not None:
                        bank_health = probe_prior_bank_health(prior_bank)
                        # prefix the P2 prior-bank scalars and histogram for
                        # the train-time channel.
                        self._log_scalar(
                            "diag_train/p2/prior_bank/mean_norm",
                            bank_health["mean_slot_norm"],
                            step,
                        )
                        self._log_scalar(
                            "diag_train/p2/prior_bank/dead_slot_fraction",
                            bank_health["dead_slot_fraction"],
                            step,
                        )
                        self._log_scalar(
                            "diag_train/p2/prior_bank/std_slot_norm",
                            bank_health["std_slot_norm"],
                            step,
                        )

                    if chm_memory is not None and prior_bank is not None:
                        p3 = probe_p3_delta(
                            prior_bank=prior_bank,
                            chm_memory=chm_memory,
                            num_prior_tokens=num_prior_tokens,
                            concat_chm_to_memory=concat_chm_to_memory,
                        )
                        self._log_p3(p3, step, prefix="diag_train")

                    # --- Slot usage histogram (I3) ------------------------
                    # Needs decoder cross-attention weights (captured by
                    # the AttentionCollector) — works for both pure-B and
                    # concat memory layouts because we slice to the first
                    # M keys.
                    xattn_names = sorted(
                        [n for n in collector.captured_names
                         if "decoder.layers." in n and "cross_attn" in n],
                        key=lambda s: int(s.split(".")[2]),
                    )
                    if xattn_names:
                        try:
                            xattn_per_layer = [collector.attn(n) for n in xattn_names]
                            slot_usage = probe_slot_usage_histogram(
                                xattn_per_layer, num_prior_tokens=num_prior_tokens
                            )
                            self._log_slot_usage(slot_usage, step, prefix="diag_train")
                        except Exception as exc:
                            logger.debug(f"skip slot-usage probe at step {step}: {exc}")
        finally:
            if was_training:
                model.train()

    # ------------------------------------------------------------------
    # Core probe pipeline
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_probes(self, trainer, pl_module) -> None:
        assert self._mha_modules is not None and self._diag_batch is not None

        model = pl_module.model
        device = pl_module.device
        was_training = model.training
        model.eval()

        imgs, chm, _gt = self._diag_batch
        imgs = imgs.to(device, non_blocking=True)
        chm = chm.to(device, non_blocking=True)

        has_prior = getattr(model, "lidar_prior", None) is not None
        concat_chm_to_memory = bool(
            getattr(model, "concat_chm_to_memory", False)
        ) if has_prior else False

        step = trainer.current_epoch

        try:
            with AttentionCollector(self._mha_modules) as collector:
                out = model(imgs, chm, return_intermediates=True)
                if isinstance(out, tuple) and len(out) >= 2:
                    _pred, intermediates = out[0], out[1]
                else:  # pragma: no cover — defensive
                    raise RuntimeError(
                        "Expected model(..., return_intermediates=True) to return a tuple."
                    )

                # --- P1 + P2 run for every captured MHA --------------------
                p1 = probe_p1_entropy(collector)
                p2 = probe_p2_norms(collector)
                p1_summary = summarise_p1(p1)
                p2_summary = summarise_p2(p2)

                self._log_p1(p1, p1_summary, step)
                self._log_p2(p2, p2_summary, step)

                # --- Attention concentration + spectral health ------------
                concentration = probe_attention_concentration(collector)
                self._log_concentration(concentration, step, prefix="diag")

                spectral = probe_decoder_v_proj_spectral(
                    self._mha_modules, filter_cross_attn=True
                )
                self._log_spectral(spectral, step, prefix="diag")

                # --- Updater spectral (I2) --------------------------------
                try:
                    updater_spec = probe_updater_spectral(model)
                    for name, stats in updater_spec.items():
                        for key in ("sigma_max", "sigma_mean", "frob", "effective_rank", "stable_rank"):
                            self._log_scalar(f"diag/spectral/{name}/{key}", stats[key], step)
                except Exception as exc:
                    logger.debug(f"skip updater-spectral probe at val ep {step}: {exc}")

                # --- CHM encoder stats (I1) -------------------------------
                chm_enc_source = intermediates.get("chm_tokens_with_pe")
                if chm_enc_source is None and not has_prior:
                    chm_enc_source = intermediates.get("chm_memory")
                if chm_enc_source is not None:
                    try:
                        chm_stats = probe_chm_encoder_stats(chm_enc_source)
                        self._log_chm_encoder(chm_stats, step, prefix="diag")
                    except Exception as exc:
                        logger.debug(f"skip CHM encoder probe at val ep {step}: {exc}")

                # --- P3 + P4 + P5 only for Plan v2 -------------------------
                p3 = None
                p4 = None
                bank_health = None
                if has_prior:
                    prior_bank = intermediates.get("prior_bank")
                    chm_memory = intermediates.get("chm_memory")
                    num_prior_tokens = int(model.lidar_prior.num_prior_tokens)

                    if prior_bank is not None:
                        bank_health = probe_prior_bank_health(prior_bank)
                        self._log_prior_bank(prior_bank, bank_health, step)

                    # Plan v2 chm_memory is [B, M, D] (pure-B) or [B, M+P', D] (concat).
                    # When the model takes the prior-only path (chm=None or chm
                    # was all-zero) chm_memory == prior broadcast → delta is 0.
                    # We still log it so you can see the prior-only delta baseline.
                    if chm_memory is not None and prior_bank is not None:
                        p3 = probe_p3_delta(
                            prior_bank=prior_bank,
                            chm_memory=chm_memory,
                            num_prior_tokens=num_prior_tokens,
                            concat_chm_to_memory=concat_chm_to_memory,
                        )
                        self._log_p3(p3, step)

                    # P4 needs the updater's attention weights; attach only if
                    # the updater was actually called this forward.
                    updater_name = "lidar_prior.updater.cross_attn"
                    if updater_name in collector.captured_names:
                        updater_attn = collector.attn(updater_name)
                        p4 = probe_p4_coverage(updater_attn)
                        self._log_p4(p4, step)

                    # --- Slot usage histogram (I3) ---------------------------
                    # Aggregated decoder→prior-slot attention mass, layer-
                    # averaged. Complements P4 (which probes the updater's
                    # attention) by probing the *decoder's* usage of the
                    # M slots — answers "how many of the M slots actually
                    # carry information the decoder uses?"
                    xattn_names = sorted(
                        [n for n in collector.captured_names
                         if "decoder.layers." in n and "cross_attn" in n],
                        key=lambda s: int(s.split(".")[2]),
                    )
                    if xattn_names:
                        try:
                            xattn_per_layer = [collector.attn(n) for n in xattn_names]
                            slot_usage = probe_slot_usage_histogram(
                                xattn_per_layer, num_prior_tokens=num_prior_tokens
                            )
                            self._log_slot_usage(slot_usage, step, prefix="diag")
                        except Exception as exc:
                            logger.debug(f"skip slot-usage probe at val ep {step}: {exc}")

                    # P5 requires concat + decoder cross-attention weights.
                    if concat_chm_to_memory:
                        # Resolve decoder cross-attn names in layer order.
                        xattn_names = sorted(
                            [n for n in collector.captured_names
                             if "decoder.layers." in n and "cross_attn" in n],
                            key=lambda s: int(s.split(".")[2]),
                        )
                        if xattn_names:
                            xattn_per_layer = [collector.attn(n) for n in xattn_names]
                            # Image grid from the intermediates.
                            if "decoder_spatial" in intermediates:
                                _, _, h_img, w_img = intermediates["decoder_spatial"].shape
                            elif "refined_tokens" in intermediates:
                                # Single-tap v2 with non-DPT head — derive from tokens.
                                B, P, _ = intermediates["refined_tokens"].shape
                                h_img = w_img = int(P ** 0.5)
                            else:
                                h_img = w_img = None

                            if h_img is not None:
                                try:
                                    p5 = probe_p5_reliance(
                                        decoder_xattn_per_layer=xattn_per_layer,
                                        num_prior_tokens=num_prior_tokens,
                                        h_img=h_img,
                                        w_img=w_img,
                                    )
                                    self._log_p5(p5, step, trainer.current_epoch)
                                except ValueError as exc:
                                    logger.debug(f"P5 skipped: {exc}")

                # --- Health flags ------------------------------------------
                flags = compute_health_flags(
                    p1_summary=p1_summary,
                    p3=p3,
                    p4=p4,
                    prior_bank_health=bank_health,
                    spectral=spectral,
                    concentration=concentration,
                )
                self._log_flags(flags, step)
        finally:
            if was_training:
                model.train()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _tb(self):
        """Return the TB SummaryWriter if present, else None."""
        writer = getattr(self._logger, "experiment", None) if self._logger else None
        if writer is not None and hasattr(writer, "add_scalar"):
            return writer
        return None

    @property
    def _logger(self):
        return self._pl_module_ref.logger if self._pl_module_ref is not None else None

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        tb = self._tb()
        if tb is not None:
            tb.add_scalar(tag, value, step)

    def _log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if not self.log_histograms:
            return
        tb = self._tb()
        if tb is None or not hasattr(tb, "add_histogram"):
            return
        try:
            tb.add_histogram(tag, values, step)
        except Exception as exc:
            logger.debug(f"skip histogram {tag}: {exc}")

    def _log_image(self, tag: str, img: torch.Tensor, step: int) -> None:
        if not self.log_images:
            return
        tb = self._tb()
        if tb is None or not hasattr(tb, "add_image"):
            return
        try:
            tb.add_image(tag, img, step)
        except Exception as exc:
            logger.debug(f"skip image {tag}: {exc}")

    def _log_p1(self, p1, p1_summary, step, prefix: str = "diag"):
        for name, per in p1.items():
            self._log_scalar(f"{prefix}/p1/{name}/mean_H", per["mean_H"], step)
            self._log_scalar(f"{prefix}/p1/{name}/mean_H_norm", per["mean_H_norm"], step)
        for k, v in p1_summary.items():
            if v != v:  # NaN check
                continue
            self._log_scalar(f"{prefix}/p1/summary/{k}", v, step)

    def _log_p2(self, p2, p2_summary, step, prefix: str = "diag"):
        for name, per in p2.items():
            self._log_scalar(f"{prefix}/p2/{name}/mean_vnorm", per["mean_vnorm"], step)
            self._log_scalar(f"{prefix}/p2/{name}/mean_kvnorm", per["mean_kvnorm"], step)
        for k, v in p2_summary.items():
            if v != v:
                continue
            self._log_scalar(f"{prefix}/p2/summary/{k}", v, step)

    def _log_prior_bank(self, prior_bank: torch.Tensor, bank_health: dict, step: int, prefix: str = "diag"):
        self._log_scalar(f"{prefix}/p2/prior_bank/mean_norm", bank_health["mean_slot_norm"], step)
        self._log_scalar(f"{prefix}/p2/prior_bank/dead_slot_fraction", bank_health["dead_slot_fraction"], step)
        self._log_scalar(f"{prefix}/p2/prior_bank/std_slot_norm", bank_health["std_slot_norm"], step)
        hist_prefix = "diag_hist" if prefix == "diag" else f"{prefix}_hist"
        self._log_histogram(
            f"{hist_prefix}/p2/prior_bank_norms",
            prior_bank.detach().norm(dim=-1).cpu(),
            step,
        )

    def _log_p3(self, p3, step, prefix: str = "diag"):
        self._log_scalar(f"{prefix}/p3/mean_delta", p3["mean_delta"], step)
        self._log_scalar(f"{prefix}/p3/std_delta", p3["std_delta"], step)
        self._log_scalar(f"{prefix}/p3/dormant_slot_fraction", p3["dormant_fraction"], step)
        self._log_scalar(f"{prefix}/p3/collapse_slot_fraction", p3["collapse_fraction"], step)
        hist_prefix = "diag_hist" if prefix == "diag" else f"{prefix}_hist"
        self._log_histogram(f"{hist_prefix}/p3/delta_per_slot", p3["delta_per_slot"], step)

    def _log_p4(self, p4, step):
        self._log_scalar("diag/p4/coverage_ratio", p4["coverage_ratio"], step)
        self._log_scalar(
            "diag/p4/mean_per_slot_entropy_norm", p4["mean_per_slot_entropy_norm"], step
        )
        self._log_scalar("diag/p4/mean_per_slot_entropy", p4["mean_per_slot_entropy"], step)
        # Log slot attention heatmap periodically (images are bulky).
        if step % self.image_log_every_n_epochs == 0:
            img = heatmap_to_rgb(p4["attn_avg"])  # [3, M, P']
            self._log_image("diag_img/p4/slot_attention_heatmap", img, step)

    def _log_p5(self, p5, step, epoch):
        self._log_scalar("diag/p5/mean_prior_reliance", p5["mean_prior_reliance"], step)
        self._log_scalar("diag/p5/median_prior_reliance", p5["median_prior_reliance"], step)
        if epoch % self.image_log_every_n_epochs == 0:
            img = reliance_map_to_rgb(p5["reliance_map"])
            self._log_image("diag_img/p5/prior_reliance_map", img, step)

    def _log_flags(self, flags, step):
        for k, v in flags.items():
            self._log_scalar(f"diag/health/{k}", float(v), step)

    def _log_concentration(self, concentration, step, prefix: str = "diag"):
        """Per-layer logit stats + max-attn weight.

        Logged as ``{prefix}/attn_conc/<name>/<metric>`` plus an aggregated
        ``{prefix}/attn_conc/decoder_summary/{max_max_attn,max_logit_abs}``.
        """
        for name, stats in concentration.items():
            short = name.replace(".mha", "")
            for key in (
                "logit_mean",
                "logit_std",
                "logit_abs_max",
                "q_norm_mean",
                "q_norm_std",
                "mean_max_attn",
                "max_max_attn",
            ):
                self._log_scalar(f"{prefix}/attn_conc/{short}/{key}", stats[key], step)

        decoder = [
            v for n, v in concentration.items()
            if n.startswith("decoder.layers.") and "cross_attn" in n
        ]
        if decoder:
            self._log_scalar(
                f"{prefix}/attn_conc/decoder_summary/max_mean_max_attn",
                max(v["mean_max_attn"] for v in decoder),
                step,
            )
            self._log_scalar(
                f"{prefix}/attn_conc/decoder_summary/max_logit_abs_max",
                max(v["logit_abs_max"] for v in decoder),
                step,
            )
            self._log_scalar(
                f"{prefix}/attn_conc/decoder_summary/max_logit_std",
                max(v["logit_std"] for v in decoder),
                step,
            )

    def _log_spectral(self, spectral, step, prefix: str = "diag"):
        """Per-layer spectral health of v_proj.

        Mirrors ``_log_train_tier_a_decoder`` but under the ``{prefix}/``
        namespace (defaults to ``diag/`` for the per-val-epoch channel).
        """
        for name, stats in spectral.items():
            short = name.replace(".cross_attn.mha", "").replace(".cross_attn", "")
            for key in ("sigma_max", "sigma_mean", "frob", "effective_rank", "stable_rank"):
                self._log_scalar(f"{prefix}/spectral/{short}/{key}", stats[key], step)

    def _log_chm_encoder(self, chm_stats: dict, step: int, prefix: str = "diag"):
        """CHM encoder output statistics (I1).

        Logs scalars under ``{prefix}/chm_enc/*``. No histograms — the
        per-token norms would be expensive at ``B * P'`` tokens and are
        summarised by mean/std/min/max already.
        """
        for key in (
            "token_norm_mean",
            "token_norm_std",
            "token_norm_min",
            "token_norm_max",
            "dead_token_fraction",
            "spatial_variance_mean",
            # Direct collapse indicators (added 2026-04-27).
            # ``mean_to_token_ratio`` and ``cos_offdiag_mean`` near 1.0 are
            # the strict signature of DC / rank-1 collapse; the historical
            # ``stable_rank`` (centred) hides this because mean-removal
            # discards the dominant DC component.
            "mean_token_norm",
            "mean_to_token_ratio",
            "cos_offdiag_mean",
            "sigma_max",
            "frob",
            # Raw spectral metrics — what we *should* have been logging
            # all along to catch DC collapse.
            "effective_rank_raw",
            "stable_rank_raw",
            # Centred metrics — residual diversity once the bank mean is
            # subtracted. Kept under explicit ``_centered`` names plus the
            # historical unqualified aliases for backwards-compat parsing
            # of older runs.
            "effective_rank_centered",
            "stable_rank_centered",
            "effective_rank",
            "stable_rank",
        ):
            if key in chm_stats:
                self._log_scalar(f"{prefix}/chm_enc/{key}", chm_stats[key], step)

    def _log_slot_usage(self, slot_usage: dict, step: int, prefix: str = "diag"):
        """Decoder → prior-slot usage histogram (I3).

        Logs scalar summaries under ``{prefix}/slot_usage/*`` plus the full
        per-slot usage histogram under ``{prefix}_hist/slot_usage/*``.
        """
        for key in (
            "entropy_norm",
            "gini",
            "top1_share",
            "top5_share",
            "active_slot_count",
        ):
            if key in slot_usage:
                self._log_scalar(f"{prefix}/slot_usage/{key}", slot_usage[key], step)

        hist_prefix = "diag_hist" if prefix == "diag" else f"{prefix}_hist"
        usage = slot_usage.get("usage_per_slot")
        if usage is not None:
            self._log_histogram(
                f"{hist_prefix}/slot_usage/usage_per_slot",
                usage,
                step,
            )

    # ------------------------------------------------------------------
    # Needed so the scalar logger helpers can reach the trainer
    # ------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        self._pl_module_ref = pl_module

    def setup(self, trainer, pl_module, stage):
        self._pl_module_ref = pl_module


# Re-export for consumers that don't want to dig into utils
__all__ = ["CHMDiagnosticsCallback"]
