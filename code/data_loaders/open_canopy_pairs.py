"""Open-Canopy multi-year (image, stale-CHM, fresh-CHM) windowed dataset.

This is **Stream C** of the domain-adaptation flat mix described in
``docs/chm/insights/B2_v1_vicreg_sreparam.md``: the only stream that
provides *real* temporal staleness in the prompt (clearcuts, regrowth,
real LiDAR scan-pattern voids, sensor reprocessing differences) rather
than synthesised staleness from :class:`CHMCorruptor`.

What a sample is
----------------
A "sample" here is a fixed-size pixel **window** that lies inside the
geographic intersection of three Open-Canopy GeoTIFFs co-registered to a
shared Lambert-93 grid (EPSG:2154, 1.5 m GSD natively):

  * ``image_year`` SPOT 6/7 pansharpened RGB+NIR tile, e.g. 2022,
  * ``prompt_year`` LiDAR-HD CHM tile, ``prompt_year < image_year``,
  * ``image_year`` LiDAR-HD CHM tile (the supervision target).

The tiles' geo-extents are read from each tile's GDAL transform so we do
not have to rely on the per-year ``.vrt`` mosaics at runtime; the
manifest builder (see :func:`build_open_canopy_pair_manifest`) does all
the geo-arithmetic offline and emits a flat list of records of the form

::

    {
      "image_year":   2022,
      "prompt_year":  2021,
      "image_path":   "canopy_height/2022/spot/compressed_pansharpened_<id_a>.tif",
      "prompt_path":  "canopy_height/2021/lidar/compressed_lidar_<id_b>.tif",
      "target_path":  "canopy_height/2022/lidar/compressed_lidar_<id_a>.tif",
      "image_window": [col, row, width, height],   # local to image_path
      "prompt_window":[col, row, width, height],   # local to prompt_path
      "target_window":[col, row, width, height],   # = image_window
    }

At runtime the dataset reads the requested windows with rasterio, packs
them into the same 5-tuple that :class:`SynRS3DHeightDataset` returns
when VICReg is enabled (``image, chm, chm_v1, chm_v2, gt_height``) so it
slots into the existing :class:`HeightEstimationModule` without any
loss-side changes, and into the flat-mix dataloader without any
collate-side changes.

Conventions matching :class:`SynRS3DHeightDataset`
--------------------------------------------------
* Image is read as 3-channel uint8 in **RGB** order. Open-Canopy SPOT
  tiles store bands ``[B, G, R, NIR]``, so we read bands ``[3, 2, 1]``
  to land RGB. Then :class:`Normalization` (``"8bit"``) divides by 255.
* CHM is read as uint16 **decimeters** and divided by 10 to land
  float32 metres — matches the height range the synthetic GT lives in.
* No-data: both prompt and target use raster value ``0``. We pass
  zeros through unchanged: in the prompt they look like the
  ``CHMCorruptor`` cutout regions B2 was trained against; in the target
  they are mostly true-ground or tile-margin pixels that B2 already
  treats as valid (``L1HeightLoss(ignore_zero=False)``). A future
  revision can promote no-data to NaN and add a mask channel to the
  return tuple, gated by a flag.
* Optional contrastive views are produced from the **target** CHM
  (clean substrate) using :class:`CHMContrastiveCorruptor` exactly like
  Stream A/B do from the GT nDSM substrate.
* Optional joint augmentation (crop / flip / spatial / color) is
  applied with :class:`Transformations` exactly like Stream A/B.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from data_loaders.synrs3d_dataset import CHMContrastiveCorruptor
from utils.normalisations import Normalization
from utils.transformations import Transformations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest schema + offline builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileGeo:
    """Minimal geo-summary of one GeoTIFF tile, in Lambert-93 (EPSG:2154).

    All Open-Canopy files use the same 1.5 m grid, so we only carry the
    top-left corner ``(x_geo, y_geo)`` and the pixel grid shape; pixel
    size is fixed at the class level.
    """

    path: str
    x_geo: float
    y_geo: float
    width: int
    height: int

    PX = 1.5  # grid resolution shared by every Open-Canopy tile

    @classmethod
    def from_path(cls, path: str | Path) -> "TileGeo":
        with rasterio.open(str(path)) as src:
            t = src.transform
            if abs(t.a - cls.PX) > 1e-3 or abs(abs(t.e) - cls.PX) > 1e-3:
                logger.warning(
                    "Tile %s has non-standard pixel size (%.3f, %.3f); expected %.2f",
                    path,
                    t.a,
                    abs(t.e),
                    cls.PX,
                )
            return cls(
                path=str(path),
                x_geo=float(t.c),
                y_geo=float(t.f),
                width=int(src.width),
                height=int(src.height),
            )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` in Lambert-93 metres."""
        x0 = self.x_geo
        x1 = self.x_geo + self.width * self.PX
        y1 = self.y_geo
        y0 = self.y_geo - self.height * self.PX  # north-up: pixel y_origin is the top
        return x0, y0, x1, y1

    def geo_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Lambert-93 (x, y) → tile-local (col, row), no clamping."""
        col = int(round((x - self.x_geo) / self.PX))
        row = int(round((self.y_geo - y) / self.PX))
        return col, row


def _tile_index(dir_path: Path) -> dict[str, TileGeo]:
    """Map ``<basename>.tif`` -> :class:`TileGeo` for every tif under ``dir_path``."""
    out: dict[str, TileGeo] = {}
    if not dir_path.is_dir():
        return out
    for tif in sorted(dir_path.iterdir()):
        if tif.suffix.lower() != ".tif":
            continue
        try:
            out[tif.name] = TileGeo.from_path(tif)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping unreadable tile %s: %s", tif, e)
    return out


def _spot_to_lidar_name(spot_name: str) -> str:
    """``compressed_pansharpened_<id>.tif`` -> ``compressed_lidar_<id>.tif``."""
    return spot_name.replace("compressed_pansharpened_", "compressed_lidar_")


def build_open_canopy_pair_manifest(
    open_canopy_root: str | Path,
    year_pairs: Iterable[tuple[int, int]] = ((2022, 2021), (2023, 2022), (2023, 2021)),
    window_size: int = 256,
    stride: int | None = None,
    image_subdir: str = "canopy_height",
    spot_dir: str = "spot",
    lidar_dir: str = "lidar",
) -> list[dict]:
    """Build the offline manifest for Stream C.

    For each ``(image_year, prompt_year)`` pair, every SPOT tile in
    ``image_year`` is intersected geographically with every LiDAR tile
    in ``prompt_year``. The intersection (in Lambert-93 metres) is then
    tiled into non-overlapping windows of size ``window_size`` × pixels
    (stride defaults to the window size).

    The supervision target is **always** ``image_year``'s LiDAR for the
    *same* SPOT-paired tile (``canopy_height/<image_year>/lidar/`` uses
    the same basenames as ``canopy_height/<image_year>/spot/``), so the
    target window is identical to the image window.

    No content-based filtering happens here: the manifest contains every
    geometrically-valid window, including ones that may be all-zero
    (tile-margin no-data). If you want to filter those out, do so as a
    post-processing pass on the returned list of records.

    Args:
        open_canopy_root:  Path to ``.../synrs3d/open_canopy/``.
        year_pairs:        Iterable of ``(image_year, prompt_year)``
                           tuples. By convention ``image_year >
                           prompt_year`` (deployment is "image is
                           newer than prompt").
        window_size:       Window edge length in pixels (1.5 m each).
                           256 px == 384 m on a side.
        stride:            Step in pixels between adjacent windows.
                           Defaults to ``window_size`` (no overlap).
        image_subdir:      Subdir under ``open_canopy_root`` containing
                           the per-year SPOT + LiDAR pair tree
                           (default ``"canopy_height"``).
        spot_dir:          Per-year SPOT subfolder name.
        lidar_dir:         Per-year LiDAR subfolder name.

    Returns:
        A list of dict records with the schema described in this
        module's docstring.
    """
    root = Path(open_canopy_root)
    if stride is None:
        stride = window_size
    if stride <= 0 or window_size <= 0:
        raise ValueError(f"window_size and stride must be positive, got {window_size}, {stride}")

    px = TileGeo.PX
    win_metres = window_size * px

    # Build per-year tile indices once.
    spot_tiles: dict[int, dict[str, TileGeo]] = {}
    lidar_tiles: dict[int, dict[str, TileGeo]] = {}
    needed_years = sorted({y for pair in year_pairs for y in pair})
    for y in needed_years:
        spot_tiles[y] = _tile_index(root / image_subdir / str(y) / spot_dir)
        lidar_tiles[y] = _tile_index(root / image_subdir / str(y) / lidar_dir)
        logger.info(
            "Open-Canopy %d: %d SPOT tiles, %d LiDAR tiles",
            y, len(spot_tiles[y]), len(lidar_tiles[y]),
        )

    records: list[dict] = []
    for image_year, prompt_year in year_pairs:
        if image_year == prompt_year:
            logger.warning("Skipping degenerate pair (%d, %d) — use Stream B for same-year.",
                           image_year, prompt_year)
            continue
        for spot_name, spot in spot_tiles[image_year].items():
            target_name = _spot_to_lidar_name(spot_name)
            target = lidar_tiles[image_year].get(target_name)
            if target is None:
                # This should never happen given the verification report
                # confirms 1:1 spot↔lidar matching within each year, but
                # we skip defensively rather than crash mid-build.
                logger.warning("No target LiDAR for SPOT %s in year %d", spot_name, image_year)
                continue

            # SPOT and same-year LiDAR usually share geo origin but may
            # differ by a pixel or two on width/height (e.g. 41596 vs
            # 41597 rows in a sample tile). We therefore *don't* assume
            # the target window equals the SPOT window — it is converted
            # through geo coords like the cross-year prompt window
            # below. That makes the dataset robust to per-tile padding
            # quirks at the cost of one extra geo→pixel round trip per
            # window during manifest build.

            sx0, sy0, sx1, sy1 = spot.bounds  # (x_min, y_min, x_max, y_max)

            for prompt_name, prompt in lidar_tiles[prompt_year].items():
                px0, py0, px1, py1 = prompt.bounds
                # Intersection in Lambert-93 metres.
                ix0 = max(sx0, px0)
                iy0 = max(sy0, py0)
                ix1 = min(sx1, px1)
                iy1 = min(sy1, py1)
                if ix1 - ix0 < win_metres or iy1 - iy0 < win_metres:
                    continue

                # Iterate window origins in geo-x (left→right) and geo-y (top→bottom).
                # Top-left corner of an x-window starts at ix0 + k*stride*px.
                gx_starts = []
                gx = ix0
                while gx + win_metres <= ix1 + 1e-6:
                    gx_starts.append(gx)
                    gx += stride * px
                gy_starts = []
                gy_top = iy1  # geo-y decreases as row increases; start from top of intersection
                while gy_top - win_metres >= iy0 - 1e-6:
                    gy_starts.append(gy_top)
                    gy_top -= stride * px

                for gx in gx_starts:
                    for gy in gy_starts:
                        s_col, s_row = spot.geo_to_pixel(gx, gy)
                        p_col, p_row = prompt.geo_to_pixel(gx, gy)
                        t_col, t_row = target.geo_to_pixel(gx, gy)
                        if not (0 <= s_col and s_col + window_size <= spot.width and
                                0 <= s_row and s_row + window_size <= spot.height):
                            continue
                        if not (0 <= p_col and p_col + window_size <= prompt.width and
                                0 <= p_row and p_row + window_size <= prompt.height):
                            continue
                        if not (0 <= t_col and t_col + window_size <= target.width and
                                0 <= t_row and t_row + window_size <= target.height):
                            continue
                        records.append({
                            "image_year":   image_year,
                            "prompt_year":  prompt_year,
                            "image_path":   str(Path(spot.path).relative_to(root)),
                            "prompt_path":  str(Path(prompt.path).relative_to(root)),
                            "target_path":  str(Path(target.path).relative_to(root)),
                            "image_window": [s_col, s_row, window_size, window_size],
                            "prompt_window":[p_col, p_row, window_size, window_size],
                            "target_window":[t_col, t_row, window_size, window_size],
                        })
        logger.info(
            "Open-Canopy pair (%d, %d): manifest now %d records",
            image_year, prompt_year, len(records),
        )

    return records


def write_manifest(records: list[dict], out_path: str | Path) -> None:
    """Persist a manifest to JSON. Uses a one-record-per-line format
    (JSONL) so very large manifests stream cheaply at load time."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps({
            "schema_version": 1,
            "n_records": len(records),
        }) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def read_manifest(path: str | Path) -> list[dict]:
    """Load a manifest written by :func:`write_manifest`."""
    p = Path(path)
    records: list[dict] = []
    with p.open("r") as f:
        header = json.loads(f.readline())
        if header.get("schema_version") != 1:
            raise ValueError(f"Unsupported manifest schema in {p}: {header}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if len(records) != header["n_records"]:
        logger.warning(
            "Manifest %s declares %d records but contains %d",
            p, header["n_records"], len(records),
        )
    return records


# ---------------------------------------------------------------------------
# Runtime dataset
# ---------------------------------------------------------------------------


class OpenCanopyStalePairDataset(Dataset):
    """Open-Canopy windowed `(image_T, CHM_{T-k}, CHM_T)` pair dataset.

    Designed to slot into the same training pipeline as
    :class:`SynRS3DHeightDataset` — same image normalisation, same
    transforms interface, same return-tuple shape — so it can be
    round-robined with the synthetic and real-clean streams via
    :class:`DAFlatMixDataset` without any loss-side or collate-side
    changes.

    Args:
        data_dir:                     Open-Canopy root, i.e.
                                      ``.../synrs3d/open_canopy``.
        manifest_path:                JSON-lines manifest produced by
                                      :func:`build_open_canopy_pair_manifest`
                                      and :func:`write_manifest`.
        normalisation:                Image normalisation method passed
                                      to :class:`Normalization` (the
                                      Stream A/B configs use
                                      ``"8bit"``).
        chm_contrastive_corruption:   Optional dict of kwargs for
                                      :class:`CHMContrastiveCorruptor`.
                                      When provided,
                                      ``__getitem__`` returns a 5-tuple
                                      ``(image, chm, chm_v1, chm_v2,
                                      gt_height)`` matching
                                      :class:`SynRS3DHeightDataset`'s
                                      VICReg-on signature. The two
                                      contrastive views are sampled
                                      from the **target** CHM (the
                                      clean substrate for this stream).
        transforms_config:            Optional dict of kwargs for
                                      :class:`Transformations`. Same
                                      semantics as for Stream A/B.
        channels:                     Number of image channels to
                                      return. Open-Canopy SPOT carries
                                      4 bands ``[B, G, R, NIR]``; we
                                      always read the RGB triplet
                                      (bands ``[3, 2, 1]``) and discard
                                      NIR. ``channels`` therefore must
                                      be 3.
        chm_corruption:               **Ignored** (signature kept for
                                      uniform YAML schema with Stream
                                      A/B). Stream C never re-corrupts
                                      a real-stale prompt — reality
                                      *is* the corruption.
        nodata_to_zero:               If True, raster values that
                                      coincide with the per-tile
                                      ``nodata`` sentinel are coerced
                                      to ``0.0`` before any downstream
                                      processing. Default True.
        year_pairs:                   Optional list of
                                      ``[image_year, prompt_year]``
                                      pairs to retain. When set, every
                                      manifest record whose
                                      ``(image_year, prompt_year)``
                                      tuple is *not* in this list is
                                      filtered out at construction
                                      time. Lets a YAML pin down the
                                      staleness gap (e.g. only
                                      ``[[2023, 2021]]`` for the
                                      cleanest 2-year-stale signal)
                                      without rebuilding the manifest.
                                      Default ``None`` keeps every
                                      record.
    """

    REQUIRED_CHANNELS = 3
    SPOT_RGB_BAND_INDEX = (3, 2, 1)  # rasterio is 1-indexed: [R, G, B] from BGRNIR
    LIDAR_DM_TO_M = 1.0 / 10.0       # Open-Canopy LiDAR-HD CHM is in decimetres

    def __init__(
        self,
        data_dir: str,
        manifest_path: str,
        normalisation: str = "8bit",
        chm_contrastive_corruption: dict | None = None,
        transforms_config: dict | None = None,
        channels: int = 3,
        chm_corruption: dict | None = None,  # accepted but ignored — see docstring
        nodata_to_zero: bool = True,
        year_pairs: list | None = None,
        **kwargs,
    ):
        super().__init__()
        if channels != self.REQUIRED_CHANNELS:
            raise ValueError(
                f"OpenCanopyStalePairDataset requires channels=3 "
                f"(SPOT RGB), got channels={channels}."
            )
        if chm_corruption is not None:
            logger.info(
                "OpenCanopyStalePairDataset: ignoring chm_corruption=%s — "
                "Stream C uses the real stale LiDAR as the prompt without "
                "additional synthetic corruption.",
                chm_corruption,
            )

        self.data_dir = Path(data_dir)
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Open-Canopy manifest not found: {self.manifest_path}. "
                f"Build it with scripts/data/build_open_canopy_manifest.py."
            )
        self.records = read_manifest(self.manifest_path)

        if year_pairs is not None:
            allowed = {(int(p[0]), int(p[1])) for p in year_pairs}
            n_before = len(self.records)
            self.records = [
                r for r in self.records
                if (r.get("image_year"), r.get("prompt_year")) in allowed
            ]
            logger.info(
                "OpenCanopyStalePairDataset: filtered by year_pairs=%s: "
                "%d -> %d records",
                sorted(allowed), n_before, len(self.records),
            )
            if not self.records:
                raise ValueError(
                    f"OpenCanopyStalePairDataset: year_pairs={sorted(allowed)} "
                    f"matched zero records in {self.manifest_path}. Check "
                    f"that the manifest contains the requested image/prompt "
                    f"year combinations."
                )

        self.normalize = Normalization(method=normalisation)
        self.nodata_to_zero = bool(nodata_to_zero)

        self.contrastive_corruptor: CHMContrastiveCorruptor | None = None
        if chm_contrastive_corruption is not None:
            self.contrastive_corruptor = CHMContrastiveCorruptor(
                **dict(chm_contrastive_corruption)
            )

        self.transformations: Transformations | None = None
        if transforms_config is not None:
            self.transformations = Transformations(**dict(transforms_config))

        logger.info(
            "OpenCanopyStalePairDataset: %d windows from %s",
            len(self.records), self.manifest_path,
        )

    # ------------------------------------------------------------------
    # Standard Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def _read_image_window(self, rel_path: str, win: list[int]) -> np.ndarray:
        """Read a 3-band RGB window from a SPOT tile as ``uint8 [3, H, W]``."""
        with rasterio.open(str(self.data_dir / rel_path)) as src:
            window = Window(col_off=int(win[0]), row_off=int(win[1]),
                             width=int(win[2]), height=int(win[3]))
            arr = src.read(indexes=list(self.SPOT_RGB_BAND_INDEX), window=window)
        if arr.shape[0] != 3:
            raise RuntimeError(
                f"Expected 3-band read from {rel_path}, got shape {arr.shape}"
            )
        return arr  # uint8

    def _read_chm_window(self, rel_path: str, win: list[int]) -> np.ndarray:
        """Read a 1-band LiDAR window in metres as ``float32 [H, W]``."""
        with rasterio.open(str(self.data_dir / rel_path)) as src:
            window = Window(col_off=int(win[0]), row_off=int(win[1]),
                             width=int(win[2]), height=int(win[3]))
            arr = src.read(1, window=window)
            nodata = src.nodata
        chm = arr.astype(np.float32) * self.LIDAR_DM_TO_M
        if self.nodata_to_zero and nodata is not None:
            chm[arr == nodata] = 0.0
        return chm

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # --- raster reads ---
        image = self._read_image_window(rec["image_path"], rec["image_window"])  # [3, H, W] uint8
        prompt = self._read_chm_window(rec["prompt_path"], rec["prompt_window"])  # [H, W] float32 metres
        target = self._read_chm_window(rec["target_path"], rec["target_window"])  # [H, W] float32 metres

        # --- normalise image (matches Stream A/B "8bit" path) ---
        image = self.normalize.apply(image)  # [C, H, W] float in [0, 1]

        # --- VICReg contrastive views from the clean substrate (target) ---
        chm_v1: np.ndarray | None = None
        chm_v2: np.ndarray | None = None
        if self.contrastive_corruptor is not None:
            chm_v1 = self.contrastive_corruptor(target)
            chm_v2 = self.contrastive_corruptor(target)

        # --- joint augmentation (same spatial op on every CHM/mask) ---
        if self.transformations is not None:
            mask_list = [prompt, target]
            if chm_v1 is not None:
                mask_list.extend([chm_v1, chm_v2])
            image, transformed = self.transformations.apply(image, mask_list)
            prompt, target = transformed[0], transformed[1]
            if chm_v1 is not None:
                chm_v1, chm_v2 = transformed[2], transformed[3]

        # --- to tensors (matches SynRS3DHeightDataset return shape) ---
        image_t = torch.as_tensor(image, dtype=torch.float32)
        prompt_t = torch.as_tensor(prompt, dtype=torch.float32)
        target_t = torch.as_tensor(target, dtype=torch.float32)
        if prompt_t.ndim == 2:
            prompt_t = prompt_t.unsqueeze(0)
        if target_t.ndim == 2:
            target_t = target_t.unsqueeze(0)

        if chm_v1 is None:
            return image_t, prompt_t, target_t

        v1 = torch.as_tensor(chm_v1, dtype=torch.float32)
        v2 = torch.as_tensor(chm_v2, dtype=torch.float32)
        if v1.ndim == 2:
            v1 = v1.unsqueeze(0)
        if v2.ndim == 2:
            v2 = v2.unsqueeze(0)
        return image_t, prompt_t, v1, v2, target_t
