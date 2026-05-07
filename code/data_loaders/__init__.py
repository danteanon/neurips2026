"""Lazy re-exports for ``data_loaders``.

The previous implementation eagerly imported every submodule (including the
parquet_tiling + label_studio chain) just to populate the public API. That
forced every consumer — including height-estimation training, which only
uses ``SynRS3DHeightDataset`` — to pay the cost of loading those subsystems
and to ship their dependencies.

This module keeps the same public API (``from data_loaders import X`` and
``getattr(data_loaders, 'X')`` both still work) but defers the submodule
import to the first attribute access, via PEP 562's module ``__getattr__``.

If you add a new class to a submodule and want it re-exported here, add an
entry to ``_SYMBOL_SOURCES``.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

# Map public symbol name -> dotted submodule it lives in (relative to this
# package). Adding a new entry is a one-line change; no import-time cost is
# incurred until someone accesses the symbol.
_SYMBOL_SOURCES: Dict[str, str] = {
    # --- data_pipeline ---
    "SegmentationDataset": "data_pipeline",
    "SegmentationDatasetPNG": "data_pipeline",
    "SegmentationDatasetFolders": "data_pipeline",
    "SegmentationDatasetFoldersPNGTiff": "data_pipeline",
    "SelectiveChannelLMDBDataset": "data_pipeline",
    "SelectiveChannelLMDBDatasetFiltered": "data_pipeline",
    "InferenceDataset": "data_pipeline",
    "FbInferenceDataset": "data_pipeline",
    "PolygonBboxDataset": "data_pipeline",
    "PolygonBboxCentroidDataset": "data_pipeline",
    "PolygonBboxDatasetSeparateFolders": "data_pipeline",
    "PolygonBboxCentroidDatasetSeparateFolders": "data_pipeline",
    # --- parquet_sync / parquet_dataset ---
    "ParquetSync": "parquet_sync",
    "ParquetTileDataset": "parquet_dataset",
    "ParquetTileDatasetWithAugmentation": "parquet_dataset",
    "StreamingParquetDataset": "parquet_dataset",
    # --- ade20k ---
    "ADE20KDataset": "ade20k_dataset",
    "ADE20KSingleClassDataset": "ade20k_dataset",
    # --- detection ---
    "DetectionDataset": "detection_dataset",
    "DetectionDatasetFolders": "detection_dataset",
    "detection_collate_fn": "detection_dataset",
    # --- height estimation ---
    "SynRS3DHeightDataset": "synrs3d_dataset",
    "OpenCanopyStalePairDataset": "open_canopy_pairs",
    "DAFlatMixDataset": "da_flat_mix",
    "PerSourceBatchDataset": "per_source_batch",
    "per_source_collate": "per_source_batch",
    "per_source_collate_split": "per_source_batch",
    "ARKitScenesDepthDataset": "arkitscenes_dataset",
    "HypersimDepthDataset": "hypersim_dataset",
}

__all__ = sorted(_SYMBOL_SOURCES.keys())


def __getattr__(name: str) -> Any:
    """Import the owning submodule on first access."""
    submod = _SYMBOL_SOURCES.get(name)
    if submod is None:
        raise AttributeError(
            f"module 'data_loaders' has no attribute {name!r} "
            f"(known re-exports: {', '.join(__all__)})"
        )
    module = importlib.import_module(f".{submod}", package=__name__)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"submodule 'data_loaders.{submod}' does not expose {name!r} — "
            f"update data_loaders/__init__.py::_SYMBOL_SOURCES"
        ) from exc
    # Cache on the package so subsequent accesses are free.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
