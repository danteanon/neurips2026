"""Health probes for the CHM-prompted height model.

Adds three diagnostics that complement :mod:`lightning_modules.chm_diagnostics_callback`:

1. **Per-module gradient norms** — at every ``log_grad_every_n_steps`` optimiser
   step, log ``‖grad‖_2`` for ``chm_encoder``, ``head`` (DPT or other),
   ``chm_recon_head`` (when present), ``chm_vicreg_expander`` (when present),
   and the decoder. Catches encoder gradient starvation in real time and lets
   us tell whether the gradient signal is reaching both the height head and the
   CHM-side parameters during a run, not only after the fact via forensic
   probes.

2. **Cross-attention weight statistics** — every ``log_attn_every_n_steps``
   training step we toggle :attr:`MultiHeadAttention.capture_attn` on a small
   subset of decoder cross-attn modules, run the next forward, and read out
   attention tensors of shape ``[B, h, T_q, T_kv]``. We log:

   * ``mean_max_attn`` — average of per-query max attention weight. Approaches
     ``1/T_kv`` when the decoder is producing uniform attention (CHM ignored)
     and approaches ``1`` when each query has a single dominant CHM target.
   * ``entropy_attn`` — Shannon entropy of attention weights, normalised to
     ``[0, 1]`` against ``log(T_kv)``. ``1`` is uniform / ignored,
     low values mean specialisation.

3. **Input-sensitivity probe** — at every ``input_sens_every_n_epochs``-th
   train epoch end, generate four synthetic CHM inputs (structured, zeros,
   constant 5 m, constant 25 m) and pass them through ``model.chm_encoder``
   only. Log pairwise cosine similarity of the mean-pooled tokens. Cosine
   close to ``1`` between distinct inputs is the input-blindness fingerprint
   (W0 epoch 0 territory).

A previously-included CHM-zeroing val probe was removed: it ran 2×N full-model
forwards every validation epoch, which on V2DPT cost ~25 min per epoch and on
V1DPT ~10 min — and on V2 it fired during sanity check, indistinguishable from
a launch hang. The same signal can be obtained much more cheaply by spot-
checking val MAE with CHM zeroed at the end of training, not every epoch.

Config schema example::

    chm_health_probes:
      enabled: true
      log_grad_every_n_steps: 50
      log_attn_every_n_steps: 200
      input_sens_every_n_epochs: 1
      attn_max_layers: 3

The callback is a no-op for non-height runs and lazily skips features for
which the model has no matching attribute (e.g. no ``chm_recon_head`` if
``aux_chm_recon: false``).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback


def _module_grad_norm(module: Optional[nn.Module]) -> Optional[float]:
    """Return ``sqrt(sum_p ‖p.grad‖²)`` over module's params, or ``None`` if
    no parameter has a populated ``.grad`` tensor.

    NaN-tolerant: NaN entries in the squared-grad sum are replaced with 0
    before reduction so a single bad gradient doesn't blank the entire
    module's signal.
    """
    if module is None:
        return None
    total: float = 0.0
    has_grad = False
    for p in module.parameters():
        if p.grad is not None:
            sq = p.grad.detach().pow(2)
            sq = torch.nan_to_num(sq, nan=0.0, posinf=0.0, neginf=0.0)
            total += float(sq.sum().item())
            has_grad = True
    return math.sqrt(total) if has_grad else None


def _find_cross_attn_mhas(model: nn.Module) -> List[nn.Module]:
    """Locate every cross-attention :class:`MultiHeadAttention` instance in
    the decoder. Works for V1/V2/V3 by name pattern: any module named
    ``cross_attn.mha`` (or ``...layers.{i}.cross_attn.mha``).
    """
    out: List[nn.Module] = []
    for name, m in model.named_modules():
        if name.endswith("cross_attn.mha"):
            out.append(m)
    return out


def _attention_stats(
    attn: torch.Tensor, eps: float = 1.0e-12
) -> tuple[float, float]:
    """Compute ``mean_max_attn`` and ``normalised_entropy`` from an attention
    tensor of shape ``[B, h, T_q, T_kv]``.

    NaN-tolerant: degenerate rows (all-NaN due to fully-masked queries or
    upstream numerical issues) are dropped before reduction. Returns ``nan``
    for both stats when every row is degenerate, but the caller can still
    log it.
    """
    T_kv = attn.shape[-1]
    flat = attn.reshape(-1, T_kv)                           # [N_rows, T_kv]
    valid = ~torch.isnan(flat).any(dim=-1)                  # [N_rows]
    flat = flat[valid]
    if flat.numel() == 0:
        return float("nan"), float("nan")
    mean_max = float(flat.amax(dim=-1).mean().item())
    p = flat.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)
    norm_entropy = float((entropy / math.log(T_kv)).mean().item())
    return mean_max, norm_entropy


def _make_synthetic_chm(
    batch_size: int, h: int, w: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build four synthetic CHM tensors used by the input-sensitivity probe."""
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, h, device=device),
        torch.linspace(0.0, 1.0, w, device=device),
        indexing="ij",
    )
    grad = (15.0 * (yy + xx)).expand(batch_size, 1, h, w).clone()
    grad = grad + 0.5 * torch.randn_like(grad)
    return {
        "real":     grad,
        "zero":     torch.zeros(batch_size, 1, h, w, device=device),
        "low_5m":   torch.full((batch_size, 1, h, w), 5.0, device=device),
        "tall_25m": torch.full((batch_size, 1, h, w), 25.0, device=device),
    }


class CHMHealthProbes(Callback):
    """See module docstring."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_grad_every_n_steps: int = 50,
        log_attn_every_n_steps: int = 200,
        input_sens_every_n_epochs: int = 1,
        attn_max_layers: int = 3,
        input_sens_image_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.log_grad_every_n_steps = int(log_grad_every_n_steps)
        self.log_attn_every_n_steps = int(log_attn_every_n_steps)
        self.input_sens_every_n_epochs = int(input_sens_every_n_epochs)
        self.attn_max_layers = int(attn_max_layers)
        self._input_sens_size = input_sens_image_size

        self._cross_attn_modules: List[nn.Module] = []
        # Indices into ``_cross_attn_modules`` that will actually have
        # ``capture_attn`` flipped on. Capping this is the difference
        # between ~1 GB and ~8 GB of extra peak memory at batch=32 on V2
        # (10 decoder layers): we only need 3 representative layers
        # (first, middle, last) to tell whether attention is uniform.
        # Computed once in ``setup``.
        self._capture_layer_indices: List[int] = []
        self._capture_attn_pending: bool = False

    # ------------------------------------------------------------------
    # Setup: locate cross-attn modules once and pick the layers we will
    # actually capture from.
    # ------------------------------------------------------------------
    def setup(self, trainer, pl_module, stage: str) -> None:
        if not self.enabled:
            return
        model = pl_module.model
        self._cross_attn_modules = _find_cross_attn_mhas(model)

        n = len(self._cross_attn_modules)
        if n == 0:
            self._capture_layer_indices = []
            return
        # Pick first, last, and (attn_max_layers - 2) evenly-spaced
        # interior layers. For n_layers <= attn_max_layers this picks
        # all layers.
        k = min(self.attn_max_layers, n)
        if k == 1:
            picks = [0]
        elif k == 2:
            picks = [0, n - 1]
        else:
            interior = [
                int(round(i * (n - 1) / (k - 1)))
                for i in range(1, k - 1)
            ]
            picks = sorted(set([0] + interior + [n - 1]))
        self._capture_layer_indices = picks

    # ------------------------------------------------------------------
    # 1. Per-module grad norms
    # ------------------------------------------------------------------
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if not self.enabled:
            return
        step = int(trainer.global_step)
        # Same first-cadence-tick guard as the attention probe — keeps the
        # callback inert during the first optimiser step so Lightning's
        # initial DDP / TB-flush / hp_metric path runs uninterrupted.
        if step < self.log_grad_every_n_steps:
            return
        if step % self.log_grad_every_n_steps != 0:
            return

        model = pl_module.model
        sources = {
            "chm_encoder":         getattr(model, "chm_encoder", None),
            "decoder":             getattr(model, "decoder", None),
            "head":                getattr(model, "head", None),
            "chm_recon_head":      getattr(model, "chm_recon_head", None),
            "vicreg_expander":     getattr(pl_module, "chm_vicreg_expander", None),
            "lidar_prior":         getattr(model, "lidar_prior", None),
        }
        for name, sub in sources.items():
            g = _module_grad_norm(sub)
            if g is not None:
                pl_module.log(
                    f"diag_train/grads/{name}_norm",
                    g,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )

    # ------------------------------------------------------------------
    # 2. Cross-attention weight statistics
    # ------------------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not self.enabled or not self._cross_attn_modules:
            return

        # Always clear leftover capture state at the start of every batch
        # so a missed ``on_train_batch_end`` (e.g. an exception in the
        # training step) cannot leave ``capture_attn`` permanently True
        # and silently retain attention tensors across thousands of
        # subsequent steps.
        for mha in self._cross_attn_modules:
            mha.capture_attn = False
            mha._last_attn = None
        self._capture_attn_pending = False

        step = int(trainer.global_step)
        # Skip the warm-up window — capturing attention on the very first
        # forward before any optimiser step has populated parameters
        # caused an OOM-retry loop on V2DPT at launch (each captured
        # attention tensor adds ~B × h × T_q × T_kv × 2 bytes, and even
        # 3 layers' worth at training-time batch sizes can push V2 over
        # the 49 GB GPU edge). Wait until the first cadence tick.
        if step < self.log_attn_every_n_steps:
            return
        if step % self.log_attn_every_n_steps != 0:
            return
        # Only enable capture on the chosen subset of layers — capturing
        # on every decoder layer was unnecessary (we only log
        # ``attn_max_layers`` of them anyway) and was the dominant
        # cause of the V2 OOM at training-time batch sizes.
        for idx in self._capture_layer_indices:
            self._cross_attn_modules[idx].capture_attn = True
        self._capture_attn_pending = True

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if not self._capture_attn_pending:
            return

        for layer_idx in self._capture_layer_indices:
            mha = self._cross_attn_modules[layer_idx]
            attn = getattr(mha, "_last_attn", None)
            if attn is None:
                mha.capture_attn = False
                continue

            try:
                mean_max, norm_entropy = _attention_stats(attn)
                pl_module.log(
                    f"diag_train/attn_chm/L{layer_idx:02d}/mean_max",
                    mean_max,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )
                pl_module.log(
                    f"diag_train/attn_chm/L{layer_idx:02d}/entropy",
                    norm_entropy,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )
            finally:
                mha._last_attn = None
                mha.capture_attn = False

        self._capture_attn_pending = False

    # ------------------------------------------------------------------
    # 3. Input-sensitivity probe
    # ------------------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not self.enabled or self.input_sens_every_n_epochs <= 0:
            return
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.input_sens_every_n_epochs != 0:
            return

        chm_encoder = getattr(pl_module.model, "chm_encoder", None)
        if chm_encoder is None:
            return

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        size = self._input_sens_size or 256
        chms = _make_synthetic_chm(2, size, size, device)
        pooled: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, c in chms.items():
                z, _, _ = chm_encoder(c)
                pooled[name] = z.mean(dim=1).mean(dim=0)         # [D]

        keys = ["real", "zero", "low_5m", "tall_25m"]
        for i, ki in enumerate(keys):
            for kj in keys[i + 1:]:
                a = pooled[ki]
                b = pooled[kj]
                cos = float(
                    (a @ b) / (a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12))
                )
                pl_module.log(
                    f"diag_train/input_sens/cos_{ki}_vs_{kj}",
                    cos,
                    sync_dist=False,
                )

        if was_training:
            pl_module.train()
