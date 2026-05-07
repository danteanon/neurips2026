"""Per-source stratified batching.

Sister of :class:`DAFlatMixDataset`, but with a *strict* per-batch composition
guarantee: every batch contains at least one sample from every sub-source.

When you wrap this in a standard :class:`torch.utils.data.DataLoader` with
``batch_size=K``, the DataLoader will materialise ``K * len(streams)`` actual
samples per batch — ``K`` from each source. So with 4 sub-streams (e.g.
SynRS3D, DFC, OGC, GeoNRW) and ``DataLoader(..., batch_size=1)`` the *real*
batch size hitting the model is 4, with one sample per source. ``batch_size=2``
gives 8, etc.

This is the right choice when you want **every gradient step** to see all
source distributions — useful when one source dominates by sample count
(GeoNRW has 122k vs DFC's 11k, so a uniform sampler would feed almost-pure
GeoNRW batches and let DFC drift over the course of an epoch).

Compare with :class:`DAFlatMixDataset` which mixes by global index ratio: it
gets the average composition right over many batches but a single small batch
might be all-source-A by chance.

Usage
-----

::

    ds = data_loaders.PerSourceBatchDataset(
        streams=[stream_a_cfg, stream_b_dfc_cfg, stream_b_ogc_cfg, stream_b_geonrw_cfg],
    )
    dl = DataLoader(
        ds,
        batch_size=2,                       # → real batch of 8 (4 sources × 2)
        shuffle=True,
        collate_fn=ds.collate_fn,           # required! see below
        num_workers=4,
    )

Notes
-----
* ``__getitem__`` returns a *list* of ``len(streams)`` samples, one per
  source. The standard PyTorch collate doesn't know what to do with that, so
  use the dataset's :py:meth:`collate_fn` (or pass the equivalent free
  function :func:`per_source_collate`) to flatten + stack.
* All sub-streams must return tuples of the same arity (3 for the standard
  ``(image, prompt, target)``, or 5 with VICReg contrastive views). Validated
  on construction.
* Logical epoch length defaults to the *longest* sub-stream so every source
  is exhausted at least once. Smaller sources wrap modulo their length.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch.utils.data import Dataset, default_collate

import data_loaders

logger = logging.getLogger(__name__)


def per_source_collate(batch):
    """Flatten per-round sample lists, then apply the default collate.

    ``batch`` is a list of length ``DataLoader.batch_size``, where each element
    is itself a list of ``len(streams)`` per-source samples (the raw output of
    :meth:`PerSourceBatchDataset.__getitem__`). We flatten in source order so
    the resulting batch tensor's leading dim is interpretable as
    ``[source_0_round_0, source_1_round_0, ..., source_0_round_1, ...]``.
    """
    flat = [sample for round_samples in batch for sample in round_samples]
    return default_collate(flat)


def per_source_collate_split(batch):
    """Collate per-round sample lists into *separate* sub-batches per source.

    Returns a **list** of collated tuples, one per source stream.  Each
    element is what ``default_collate`` would produce for the samples from
    that stream alone (i.e. a tuple of tensors with leading dim =
    ``DataLoader.batch_size``).

    Usage in config::

        dataloader:
          collate_fn: "per_source_collate_split"

    The lightning module's ``training_step`` receives the list and can
    apply a different loss recipe per sub-batch (per-source loss routing).
    """
    n_sources = len(batch[0])
    sub_batches = []
    for src_idx in range(n_sources):
        source_samples = [round_samples[src_idx] for round_samples in batch]
        sub_batches.append(default_collate(source_samples))
    return sub_batches


class PerSourceBatchDataset(Dataset):
    """Wrap N sub-datasets so every batch contains ≥1 sample per source.

    Args:
        data_dir:        Unused (kept positional for compatibility with
                         ``SegmentationDataModule``'s factory which pops
                         ``data_dir`` before forwarding kwargs). Each
                         sub-stream config carries its own ``data_dir``.
        streams:         Ordered list of sub-stream configs. Each entry is a
                         dict with ``class`` and ``data_dir`` plus arbitrary
                         kwargs forwarded to that class's constructor.
                         ``transforms`` is renamed to ``transforms_config`` to
                         match the existing dataset constructors.
        length:          Logical epoch length. Defaults to the *longest* sub-
                         stream so every source is exhausted at least once.
                         Increase to oversample across many epochs in one
                         logical pass.
        transforms_config: Ignored — top-level transforms must live in each
                         sub-stream config.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        streams: Sequence[dict] | None = None,
        length: int | None = None,
        transforms_config=None,  # accepted for factory compat, intentionally ignored
        **kwargs,
    ):
        super().__init__()
        del data_dir, transforms_config, kwargs

        if not streams:
            raise ValueError(
                "PerSourceBatchDataset: 'streams' must list at least one sub-stream"
            )

        self.sub_datasets: list[Dataset] = []
        self.stream_names: list[str] = []
        for i, cfg in enumerate(streams):
            cfg = dict(cfg)
            class_name = cfg.pop("class", None)
            if class_name is None:
                raise ValueError(f"PerSourceBatchDataset: stream {i} missing 'class'")
            stream_data_dir = cfg.pop("data_dir", None)
            if stream_data_dir is None:
                raise ValueError(
                    f"PerSourceBatchDataset: stream {i} ({class_name}) missing 'data_dir'"
                )
            transforms = cfg.pop("transforms", None)
            if transforms is not None:
                cfg["transforms_config"] = transforms
            label = cfg.pop("name", None) or class_name
            ds_cls = getattr(data_loaders, class_name)
            ds = ds_cls(stream_data_dir, **cfg)
            self.sub_datasets.append(ds)
            self.stream_names.append(f"S{i}:{label}")
            logger.info(
                "PerSourceBatchDataset stream %d (%s): %d samples",
                i, label, len(ds),
            )

        # Default logical epoch: longest stream so every source is fully
        # covered at least once per epoch (smaller streams wrap).
        if length is None:
            length = max(len(d) for d in self.sub_datasets)
        self.length = int(length)

        # Validate every stream returns the same tuple shape — the per-round
        # list must be uniform so default_collate can stack them.
        sample_shapes = [len(d[0]) for d in self.sub_datasets]
        if len(set(sample_shapes)) != 1:
            raise RuntimeError(
                f"PerSourceBatchDataset: sub-streams return tuples of different "
                f"sizes {dict(zip(self.stream_names, sample_shapes))}. Make sure "
                f"all streams agree on whether contrastive views are produced "
                f"(configure chm_contrastive_corruption identically across streams)."
            )

        logger.info(
            "PerSourceBatchDataset: n_sources=%d logical_len=%d "
            "tuple_arity=%d effective_min_batch=DataLoader.batch_size×%d",
            len(self.sub_datasets), self.length, sample_shapes[0],
            len(self.sub_datasets),
        )

    # ------------------------------------------------------------------
    # Standard Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> list:
        if idx < 0:
            idx += self.length
        if not (0 <= idx < self.length):
            raise IndexError(idx)
        # Each sub-dataset wraps modulo its own length, so smaller sources
        # repeat-cycle until the longest finishes.
        return [sub[idx % len(sub)] for sub in self.sub_datasets]

    # ------------------------------------------------------------------
    # Collate hook
    # ------------------------------------------------------------------
    @property
    def collate_fn(self):
        """Convenient handle for ``DataLoader(collate_fn=ds.collate_fn)``.

        Returns :func:`per_source_collate` (flat) by default. Switch to
        ``per_source_collate_split`` in the config for per-source loss routing.
        """
        return per_source_collate

    @property
    def n_sources(self) -> int:
        return len(self.sub_datasets)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stream_summary(self) -> list[dict]:
        return [
            {
                "name": name,
                "n_samples": len(ds),
                "class": type(ds).__name__,
            }
            for name, ds in zip(self.stream_names, self.sub_datasets)
        ]
