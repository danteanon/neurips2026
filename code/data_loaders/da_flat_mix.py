"""Flat-mix wrapper for the height-estimation domain-adaptation pipeline.

Composes three sub-datasets (each one already returning the standard
``(image, chm, [chm_v1, chm_v2,] gt_height)`` tuple expected by
:class:`HeightEstimationModule`) into a single :class:`Dataset` that
yields samples in a fixed integer ratio per logical step.

The streams, in the canonical order documented in
``docs/chm/insights/B2_v1_vicreg_sreparam.md``:

  Stream A — synthetic SynRS3D + :class:`CHMCorruptor`
            (synthesised stale prompt; instantiated as
            :class:`SynRS3DHeightDataset` pointed at the synthetic
            subsets).
  Stream B — Tier-A real-clean (DFC, OGC, GeoNRW) +
            :class:`CHMCorruptor`. Same class as Stream A, just a
            different ``data_dir`` + ``subsets`` list.
  Stream C — Open-Canopy multi-year windowed pairs from
            :class:`OpenCanopyStalePairDataset` — the only stream where
            the prompt is genuinely stale LiDAR (T-k years old).

How "flat mix" works
--------------------
Given ratios ``[a, b, c]``, the mixer builds a deterministic interleave
plan of length ``a + b + c`` (e.g. ``[A, A, B, C]`` for ``[2, 1, 1]``)
and indexes into it modulo with the global sample index. The chosen
stream's index is then ``i // sum(ratios)`` modulo that stream's
length, so each stream is iterated in order and wraps around as
needed. This makes one logical "epoch" of length ``length`` a
deterministic sweep over every stream in the configured ratio, which
keeps batch composition stable across epochs and across DDP ranks.

Notes
-----
* Streams must all return tuples of the **same length** (3 or 5). This
  is the case as long as ``chm_contrastive_corruption`` is configured
  (or omitted) consistently across all three sub-datasets. The mixer
  validates this on first sample and raises early.
* :class:`torch.utils.data.DataLoader` shuffling will permute the
  global indices, but because the per-stream pointer is derived from
  ``i // sum(ratios)``, the modulo arithmetic still respects the
  configured ratio across the *batch* in expectation. If perfect
  per-batch ratio is required (e.g. for separate per-stream losses),
  use a :class:`torch.utils.data.WeightedRandomSampler` and disable
  shuffle, or set ``shuffle=False`` and rely on the deterministic
  interleave plan as-is.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch.utils.data import Dataset

import data_loaders

logger = logging.getLogger(__name__)


class DAFlatMixDataset(Dataset):
    """Flat-ratio mixer for source / target-clean / target-stale streams.

    Args:
        data_dir:        Unused (kept positional for compatibility with
                         :class:`SegmentationDataModule`'s factory which
                         pops ``data_dir`` before forwarding kwargs).
                         Each sub-stream config carries its own
                         ``data_dir``.
        streams:         Ordered list of sub-stream configs. Each entry
                         is a dict with at least ``class`` and
                         ``data_dir`` plus arbitrary kwargs forwarded
                         to that class's constructor. ``transforms`` is
                         renamed to ``transforms_config`` to match the
                         existing dataset constructors.
        ratios:          Integer per-stream weights, in the same order
                         as ``streams``. The default ``[2, 1, 1]``
                         corresponds to "Stream A : Stream B : Stream
                         C = 2 : 1 : 1" — keeps source supervision
                         dense to anchor the loss recipe while the
                         real-clean and real-stale streams pull the
                         model toward the deployment distribution.
        length:          Logical epoch length. Defaults to the number
                         of unique sub-stream samples per cycle
                         multiplied by the smallest weight that
                         produces ``≥ sum(stream_lengths)`` samples,
                         which is roughly the same number of unique
                         samples seen per epoch as a naive
                         ``ConcatDataset``.
        transforms_config: Ignored — top-level transforms must live in
                         each sub-stream config.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        streams: Sequence[dict] | None = None,
        ratios: Sequence[int] = (2, 1, 1),
        length: int | None = None,
        transforms_config=None,  # accepted for factory compat, intentionally ignored
        **kwargs,
    ):
        super().__init__()
        del data_dir, transforms_config, kwargs  # see docstring

        if not streams:
            raise ValueError("DAFlatMixDataset: 'streams' must list at least one sub-stream")
        ratios = list(int(r) for r in ratios)
        if len(ratios) != len(streams):
            raise ValueError(
                f"DAFlatMixDataset: ratios ({ratios}) must match number of "
                f"streams ({len(streams)})"
            )
        if any(r < 0 for r in ratios) or sum(ratios) == 0:
            raise ValueError(f"DAFlatMixDataset: ratios must be non-negative and sum > 0, got {ratios}")

        # Build sub-datasets from configs.
        self.sub_datasets: list[Dataset] = []
        self.stream_names: list[str] = []
        for i, cfg in enumerate(streams):
            cfg = dict(cfg)
            class_name = cfg.pop("class", None)
            if class_name is None:
                raise ValueError(f"DAFlatMixDataset: stream {i} missing 'class'")
            stream_data_dir = cfg.pop("data_dir", None)
            if stream_data_dir is None:
                raise ValueError(f"DAFlatMixDataset: stream {i} ({class_name}) missing 'data_dir'")
            transforms = cfg.pop("transforms", None)
            if transforms is not None:
                cfg["transforms_config"] = transforms
            ds_cls = getattr(data_loaders, class_name)
            ds = ds_cls(stream_data_dir, **cfg)
            self.sub_datasets.append(ds)
            self.stream_names.append(f"S{i}:{class_name}")
            logger.info(
                "DAFlatMixDataset stream %d (%s, ratio %d): %d samples",
                i, class_name, ratios[i], len(ds),
            )

        self.ratios = ratios
        self.cycle_len = sum(self.ratios)

        # Build the deterministic interleave plan. For ratios=[2,1,1] this
        # yields [0, 0, 1, 2] — i.e. on every 4 sequential global indices,
        # 2 come from stream 0, 1 from stream 1, 1 from stream 2.
        self._plan: list[int] = []
        for i, r in enumerate(self.ratios):
            self._plan.extend([i] * r)

        # Default logical epoch: ~one pass over the union of streams.
        if length is None:
            length = sum(len(d) for d in self.sub_datasets)
        self.length = int(length)

        # Validate every stream returns the same tuple shape.
        sample_shapes = [len(d[0]) for d in self.sub_datasets]
        if len(set(sample_shapes)) != 1:
            raise RuntimeError(
                f"DAFlatMixDataset: sub-streams return tuples of different sizes "
                f"{dict(zip(self.stream_names, sample_shapes))}. Make sure they all "
                f"agree on whether contrastive views are produced (configure "
                f"chm_contrastive_corruption identically across streams)."
            )
        logger.info(
            "DAFlatMixDataset: ratios=%s plan=%s cycle_len=%d total_unique=%d logical_len=%d "
            "tuple_arity=%d",
            self.ratios, self._plan, self.cycle_len,
            sum(len(d) for d in self.sub_datasets), self.length, sample_shapes[0],
        )

    # ------------------------------------------------------------------
    # Standard Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.length
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        stream_idx = self._plan[idx % self.cycle_len]
        sub = self.sub_datasets[stream_idx]
        # Local pointer: sweep each stream in order, wrapping when it runs out.
        # Each stream advances once per ratio[i] global steps, so the local
        # offset within the stream is (global_idx // cycle_len) * ratio_i +
        # (position_within_cycle_for_this_stream). The simpler equivalent
        # below preserves "every sample is reachable" without juggling
        # per-stream positions explicitly.
        local = idx // self.cycle_len * self.ratios[stream_idx] + \
                self._plan[: idx % self.cycle_len].count(stream_idx)
        return sub[local % len(sub)]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stream_summary(self) -> list[dict]:
        """Return a small dict per stream for logging / debugging."""
        return [
            {
                "name": name,
                "ratio": ratio,
                "n_samples": len(ds),
                "class": type(ds).__name__,
            }
            for name, ratio, ds in zip(self.stream_names, self.ratios, self.sub_datasets)
        ]
