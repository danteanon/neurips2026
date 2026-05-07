"""Unified metric computation for height/depth benchmarking.

Every model adapter feeds (pred, gt) arrays into ``evaluate()`` and gets
back a flat dict of metrics.  The same function is used for every model,
dataset, and corruption regime so the numbers are always comparable.

Two interfaces are exposed:

- ``evaluate`` / ``evaluate_batch``: legacy API that takes lists of arrays.
  Computes per-sample metrics and aggregates with mean.  Convenient for
  small batches but holds everything in memory.

- ``StreamingMetrics``: incremental aggregator that keeps O(1) state per
  metric (running sums) instead of storing per-sample arrays.  Use this
  for large corruption sweeps where storing N×H×W float32 predictions
  per regime would blow up RAM.  Pixel-weighted aggregation is exact;
  Pearson R is computed from running cross-product sums.
"""

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr


def _sobel_magnitude(arr: np.ndarray) -> np.ndarray:
    sx = ndimage.sobel(arr, axis=0)
    sy = ndimage.sobel(arr, axis=1)
    return np.hypot(sx, sy)


def evaluate(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    canopy_threshold: float = 2.0,
) -> dict:
    """Compute standard height-estimation metrics.

    Parameters
    ----------
    pred, gt : (H, W) float32 arrays in metres.
    mask     : optional (H, W) bool — True = valid pixel.
    canopy_threshold : height (m) separating ground from canopy.

    Returns
    -------
    dict with keys:
        mae, rmse, bias,
        mae_canopy, mae_ground, n_canopy, n_ground,
        pearson_r,
        gradient_mae
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    mask = mask & np.isfinite(pred) & np.isfinite(gt)

    p = pred[mask]
    g = gt[mask]

    if p.size == 0:
        return {k: float("nan") for k in [
            "mae", "rmse", "bias",
            "mae_canopy", "mae_ground", "n_canopy", "n_ground",
            "pearson_r", "gradient_mae",
        ]}

    diff = p - g
    abs_diff = np.abs(diff)

    canopy_mask = g > canopy_threshold
    ground_mask = ~canopy_mask

    # Gradient error (Sobel on full image, then mask)
    grad_pred = _sobel_magnitude(pred)
    grad_gt = _sobel_magnitude(gt)
    grad_diff = np.abs(grad_pred - grad_gt)
    grad_mae = float(np.mean(grad_diff[mask]))

    # Pearson r (guard against constant arrays)
    if np.std(p) < 1e-8 or np.std(g) < 1e-8:
        pr = 0.0
    else:
        pr, _ = pearsonr(p, g)

    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
        "mae_canopy": float(np.mean(abs_diff[canopy_mask])) if canopy_mask.any() else float("nan"),
        "mae_ground": float(np.mean(abs_diff[ground_mask])) if ground_mask.any() else float("nan"),
        "n_canopy": int(canopy_mask.sum()),
        "n_ground": int(ground_mask.sum()),
        "pearson_r": float(pr),
        "gradient_mae": grad_mae,
    }


def evaluate_batch(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    class_masks: list[np.ndarray | None] | None = None,
    canopy_threshold: float = 2.0,
    tree_class: int = 5,
    ground_class: int = 2,
) -> dict:
    """Aggregate metrics over a batch of samples.

    Standard metrics (``mae``, ``rmse``, etc.) average per-sample values
    (legacy behaviour).  When ``class_masks`` are supplied (per-pixel
    ASPRS LiDAR class arrays — same convention as DFC2019 / Open-Canopy),
    additional pixel-weighted, class-restricted metrics are reported:

    - ``mae_tree_only``    : MAE over pixels where class == ``tree_class``
    - ``mae_ground_only``  : MAE over pixels where class == ``ground_class``
    - ``mae_tree_ground``  : MAE over pixels where class in {tree, ground}
    - ``mae_all_pixels``   : pixel-weighted MAE over the full valid mask
                             (use this instead of the per-sample-mean ``mae``
                             when comparing across heterogeneous tiles).
    """
    if masks is None:
        masks = [None] * len(preds)
    if class_masks is None:
        class_masks = [None] * len(preds)

    results = [
        evaluate(p, g, m, canopy_threshold)
        for p, g, m in zip(preds, gts, masks)
    ]

    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results
                if not (isinstance(r[k], float) and np.isnan(r[k]))]
        if k.startswith("n_"):
            agg[k] = sum(vals)
        elif vals:
            agg[k] = float(np.mean(vals))
        else:
            agg[k] = float("nan")

    # Pixel-weighted, class-restricted metrics. Walk all samples once
    # accumulating sums so we get an exact pixel-weighted mean.
    sums = {
        "all":    {"abs": 0.0, "sq": 0.0, "diff": 0.0, "n": 0},
        "tree":   {"abs": 0.0, "sq": 0.0, "diff": 0.0, "n": 0},
        "ground": {"abs": 0.0, "sq": 0.0, "diff": 0.0, "n": 0},
        "tg":     {"abs": 0.0, "sq": 0.0, "diff": 0.0, "n": 0},
    }
    n_with_classes = 0
    for p, g, m, cm in zip(preds, gts, masks, class_masks):
        p = p.astype(np.float64)
        g = g.astype(np.float64)
        valid = np.isfinite(p) & np.isfinite(g)
        if m is not None:
            valid &= m
        if not valid.any():
            continue
        diff = p - g

        def _accum(key, sel):
            sums[key]["abs"]  += float(np.abs(diff[sel]).sum())
            sums[key]["sq"]   += float((diff[sel] ** 2).sum())
            sums[key]["diff"] += float(diff[sel].sum())
            sums[key]["n"]    += int(sel.sum())

        _accum("all", valid)
        if cm is not None:
            n_with_classes += 1
            tree_sel   = valid & (cm == tree_class)
            ground_sel = valid & (cm == ground_class)
            tg_sel     = valid & ((cm == tree_class) | (cm == ground_class))
            if tree_sel.any():   _accum("tree",   tree_sel)
            if ground_sel.any(): _accum("ground", ground_sel)
            if tg_sel.any():     _accum("tg",     tg_sel)

    def _finalise(key, prefix):
        s = sums[key]
        n = s["n"]
        if n == 0:
            return {
                f"mae_{prefix}":  float("nan"),
                f"rmse_{prefix}": float("nan"),
                f"bias_{prefix}": float("nan"),
                f"n_{prefix}":    0,
            }
        return {
            f"mae_{prefix}":  s["abs"]  / n,
            f"rmse_{prefix}": float(np.sqrt(s["sq"] / n)),
            f"bias_{prefix}": s["diff"] / n,
            f"n_{prefix}":    n,
        }

    agg.update(_finalise("all",    "all_pixels"))
    agg.update(_finalise("tree",   "tree_only"))
    agg.update(_finalise("ground", "ground_only"))
    agg.update(_finalise("tg",     "tree_ground"))

    agg["n_samples"] = len(results)
    agg["n_samples_with_classes"] = n_with_classes
    return agg


class StreamingMetrics:
    """Incremental aggregator for height/depth metrics.

    Holds running sums per pixel-class subset (all / tree / ground /
    tree+ground) and per-sample sums for legacy per-sample-mean metrics.
    Memory is O(1) — no per-sample arrays are stored.
    """

    def __init__(self,
                 tree_class: int = 5,
                 ground_class: int = 2,
                 canopy_threshold: float = 2.0):
        self.tree_class = tree_class
        self.ground_class = ground_class
        self.canopy_threshold = canopy_threshold

        # Per-pixel running sums for each subset (all / tree / ground / tg)
        self._sums = {
            k: {"abs": 0.0, "sq": 0.0, "diff": 0.0,
                "p_sum": 0.0, "g_sum": 0.0,
                "p2_sum": 0.0, "g2_sum": 0.0, "pg_sum": 0.0,
                "n": 0}
            for k in ("all", "tree", "ground", "tg")
        }
        # Per-sample running stats (for legacy per-sample-mean metrics)
        self._n_samples = 0
        self._n_samples_with_classes = 0
        self._per_sample_mae_sum = 0.0
        self._per_sample_rmse_sum = 0.0
        self._per_sample_bias_sum = 0.0
        self._per_sample_canopy_mae_sum = 0.0
        self._per_sample_canopy_n = 0
        self._per_sample_ground_mae_sum = 0.0
        self._per_sample_ground_n = 0
        self._gradient_mae_sum = 0.0

    @staticmethod
    def _accumulate(bucket, p_sel, g_sel, d_sel):
        bucket["abs"]    += float(np.abs(d_sel).sum())
        bucket["sq"]     += float((d_sel ** 2).sum())
        bucket["diff"]   += float(d_sel.sum())
        bucket["p_sum"]  += float(p_sel.sum())
        bucket["g_sum"]  += float(g_sel.sum())
        bucket["p2_sum"] += float((p_sel ** 2).sum())
        bucket["g2_sum"] += float((g_sel ** 2).sum())
        bucket["pg_sum"] += float((p_sel * g_sel).sum())
        bucket["n"]      += int(d_sel.size)

    def update(self,
               pred: np.ndarray,
               gt: np.ndarray,
               class_mask: np.ndarray | None = None,
               valid_mask: np.ndarray | None = None) -> None:
        """Add a single sample's (pred, gt) — both ``(H, W)`` float arrays."""
        p = pred.astype(np.float64)
        g = gt.astype(np.float64)
        valid = np.isfinite(p) & np.isfinite(g)
        if valid_mask is not None:
            valid &= valid_mask
        self._n_samples += 1
        if not valid.any():
            return

        d = p - g
        # All-pixels accumulation
        self._accumulate(self._sums["all"], p[valid], g[valid], d[valid])

        # Per-sample legacy metrics
        v_n = int(valid.sum())
        self._per_sample_mae_sum  += float(np.abs(d[valid]).sum()) / v_n
        self._per_sample_rmse_sum += float(np.sqrt((d[valid] ** 2).mean()))
        self._per_sample_bias_sum += float(d[valid].sum()) / v_n

        canopy_sel = valid & (g > self.canopy_threshold)
        ground_sel = valid & (g <= self.canopy_threshold)
        if canopy_sel.any():
            self._per_sample_canopy_mae_sum += float(np.abs(d[canopy_sel]).mean())
            self._per_sample_canopy_n += 1
        if ground_sel.any():
            self._per_sample_ground_mae_sum += float(np.abs(d[ground_sel]).mean())
            self._per_sample_ground_n += 1

        # Gradient MAE (per-sample mean)
        sx_p = ndimage.sobel(p, axis=0); sy_p = ndimage.sobel(p, axis=1)
        sx_g = ndimage.sobel(g, axis=0); sy_g = ndimage.sobel(g, axis=1)
        gp = np.hypot(sx_p, sy_p)
        gg = np.hypot(sx_g, sy_g)
        self._gradient_mae_sum += float(np.abs(gp - gg)[valid].mean())

        # Per-class accumulation (only when LULC is available)
        if class_mask is not None:
            self._n_samples_with_classes += 1
            cm = class_mask
            tree_sel   = valid & (cm == self.tree_class)
            ground_sel = valid & (cm == self.ground_class)
            tg_sel     = valid & ((cm == self.tree_class) | (cm == self.ground_class))
            for key, sel in (("tree", tree_sel),
                             ("ground", ground_sel),
                             ("tg", tg_sel)):
                if sel.any():
                    self._accumulate(self._sums[key], p[sel], g[sel], d[sel])

    def _pixel_metrics(self, key: str, prefix: str) -> dict:
        s = self._sums[key]
        n = s["n"]
        if n == 0:
            return {
                f"mae_{prefix}":  float("nan"),
                f"rmse_{prefix}": float("nan"),
                f"bias_{prefix}": float("nan"),
                f"n_{prefix}":    0,
            }
        # Pearson r from running cross-product sums
        var_p = s["p2_sum"] / n - (s["p_sum"] / n) ** 2
        var_g = s["g2_sum"] / n - (s["g_sum"] / n) ** 2
        cov   = s["pg_sum"] / n - (s["p_sum"] * s["g_sum"]) / (n * n)
        if var_p <= 1e-12 or var_g <= 1e-12:
            pr = 0.0
        else:
            pr = float(cov / np.sqrt(var_p * var_g))
        return {
            f"mae_{prefix}":     s["abs"]  / n,
            f"rmse_{prefix}":    float(np.sqrt(s["sq"] / n)),
            f"bias_{prefix}":    s["diff"] / n,
            f"pearson_{prefix}": pr,
            f"n_{prefix}":       n,
        }

    def finalise(self) -> dict:
        n = self._n_samples
        if n == 0:
            return {"n_samples": 0}

        out: dict = {
            # Legacy per-sample-mean metrics (kept for backward compatibility)
            "mae":          self._per_sample_mae_sum  / n,
            "rmse":         self._per_sample_rmse_sum / n,
            "bias":         self._per_sample_bias_sum / n,
            "mae_canopy":   (self._per_sample_canopy_mae_sum / self._per_sample_canopy_n
                             if self._per_sample_canopy_n else float("nan")),
            "mae_ground":   (self._per_sample_ground_mae_sum / self._per_sample_ground_n
                             if self._per_sample_ground_n else float("nan")),
            "n_canopy":     self._sums["all"]["n"],  # kept for schema; not exact
            "n_ground":     self._sums["all"]["n"],
            "gradient_mae": self._gradient_mae_sum / n,
            "n_samples":    n,
            "n_samples_with_classes": self._n_samples_with_classes,
        }
        out.update(self._pixel_metrics("all",    "all_pixels"))
        out.update(self._pixel_metrics("tree",   "tree_only"))
        out.update(self._pixel_metrics("ground", "ground_only"))
        out.update(self._pixel_metrics("tg",     "tree_ground"))
        return out
