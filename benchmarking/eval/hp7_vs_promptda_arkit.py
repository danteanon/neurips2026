"""Hp7 vs PromptDA — head-to-head ARKitScenes prompt-corruption sweep.

This is the entry point for the *prompt-trust collapse* experiment in
section "Generalisation beyond aerial CHM" of the paper. We feed both
models the same RGB frame and the same metric depth prompt, then
gradually corrupt the prompt (shift ±N px, contiguous cutout area) and
report MAE on the highres-LiDAR ground truth.

Both models output metric depth in metres on the same per-frame grid:

* **Hp7** (`code/model/dinov2_height_model.py::Dinov2HeightModelDPT`) —
  our DINOv2-ViT-L + DA-V2-DPT model with cross-attention CHM fusion,
  trained on HyperSim only. It expects per-sample min-max-normalised
  prompt + sigmoid-bounded output, then we denormalise externally with
  the **clean** prompt's (min, scale) so the metre-space MAE is
  comparable across regimes.
* **PromptDA** — the published ``depth-anything/prompt-depth-anything-vitl``
  baseline. Internal min-max normalisation; we just hand it the metric
  prompt + RGB.

Reproducibility
---------------
For every (frame_id, regime) pair we seed numpy/random with
``hash((frame_id, regime))`` so cutout placement and shift directions are
identical across model passes; deltas are paired.

Hardware
--------
A 24 GB CUDA GPU is required. Set ``CUDA_VISIBLE_DEVICES`` or pass
``--device cuda:N``.

Usage
-----
::

    bash benchmarking/run_hp7_arkit_benchmark.sh
    # or directly:
    python benchmarking/eval/hp7_vs_promptda_arkit.py \\
        --hp7_ckpt weights/Hp7_hypersim_last.ckpt \\
        --data_dir data/arkitscenes/upsampling \\
        --max_total_frames 2000 \\
        --device cuda:0 \\
        --out_summary benchmarking/results/arkitscenes/results_hp7_vs_promptda_arkit_summary.json \\
        --out_markdown benchmarking/results/arkitscenes/results_hp7_vs_promptda_arkit_summary.md \\
        --out_json benchmarking/results/arkitscenes/results_hp7_vs_promptda_arkit_full.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---- path bootstrap -----------------------------------------------------
# This script lives at ``benchmarking/eval/hp7_vs_promptda_arkit.py``.
# The bundled training repo subset is at ``../code/`` relative to the
# package root. PromptDA is cloned by ``scripts/clone_competitor_repos.sh``
# into ``benchmarking/repos/PromptDA``.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PACKAGE_ROOT / "code"
PROMPTDA_ROOT = PACKAGE_ROOT / "benchmarking" / "repos" / "PromptDA"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(PROMPTDA_ROOT))

from data_loaders.synrs3d_dataset import CHMCorruptor  # noqa: E402
from lightning_modules.height_module import HeightEstimationModule  # noqa: E402
from model.get_model import get_model  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
log = logging.getLogger("hp7_vs_promptda")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE = 448  # divisible by 14 (DINOv2 patch). Both models trained / eval'd here.
MM_TO_M = 1.0 / 1000.0
MAX_DEPTH_M = 10.0


# --------------------------------------------------------------------------
# Dataset walking
# --------------------------------------------------------------------------

def _list_all_validation_frames(val_dir: Path, max_total: int | None = None):
    """Return up to ``max_total`` triplets sampled across all videos.

    Round-robin across videos so a 2000-frame budget gives near-uniform
    coverage of all 287 ARKitScenes upsampling-Validation videos rather
    than dumping the first 100 videos and ignoring the rest.
    """
    per_video: list[list[tuple[str, Path, Path, Path]]] = []
    video_dirs = sorted(d for d in val_dir.iterdir() if d.is_dir())
    for vdir in video_dirs:
        color_dir = vdir / "wide"
        if not color_dir.is_dir():
            color_dir = vdir / "color"
        lowres_dir = vdir / "lowres_depth"
        highres_dir = vdir / "highres_depth"
        if not all(d.is_dir() for d in [color_dir, lowres_dir, highres_dir]):
            continue
        triplets = []
        for cfile in sorted(color_dir.glob("*.png")):
            lr = lowres_dir / cfile.name
            hr = highres_dir / cfile.name
            if lr.exists() and hr.exists():
                triplets.append((vdir.name, cfile, lr, hr))
        if triplets:
            per_video.append(triplets)

    n_videos = len(per_video)
    total_frames = sum(len(v) for v in per_video)
    log.info("Validation set: %d videos with all 3 modalities, "
             "%d total frames", n_videos, total_frames)

    if max_total is None or max_total >= total_frames:
        flat = [t for vid_list in per_video for t in vid_list]
        return flat

    out = []
    cursors = [0] * n_videos
    while len(out) < max_total:
        progressed = False
        for i in range(n_videos):
            if cursors[i] < len(per_video[i]):
                out.append(per_video[i][cursors[i]])
                cursors[i] += 1
                progressed = True
                if len(out) >= max_total:
                    break
        if not progressed:
            break
    return out


def _load_frame(color_path: Path, lowres_path: Path, highres_path: Path,
                tile: int = TILE):
    img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_u8 = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)

    hr_raw = cv2.imread(str(highres_path), cv2.IMREAD_UNCHANGED)
    gt_full = np.clip(hr_raw.astype(np.float32) * MM_TO_M, 0, MAX_DEPTH_M)
    gt = cv2.resize(gt_full, (tile, tile), interpolation=cv2.INTER_NEAREST)

    lr_raw = cv2.imread(str(lowres_path), cv2.IMREAD_UNCHANGED)
    lr_depth = np.clip(lr_raw.astype(np.float32) * MM_TO_M, 0, MAX_DEPTH_M)
    chm = cv2.resize(lr_depth, (tile, tile), interpolation=cv2.INTER_LINEAR)
    return image_u8, chm.astype(np.float32), gt.astype(np.float32)


# --------------------------------------------------------------------------
# Regimes (operate on metric prompt; each model handles its own normalisation)
# --------------------------------------------------------------------------

def _make_shift_corruptor(max_shift: int) -> CHMCorruptor:
    return CHMCorruptor(
        cutout_prob=0.0, max_shift=max_shift, resolution_factor=1.0,
        gaussian_blur_sigma=0.0, height_noise_sigma=0.0,
        full_dropout_prob=0.0, always_on_noise_sigma=0.0,
    )


def _make_cutout_corruptor(area: float) -> CHMCorruptor:
    return CHMCorruptor(
        cutout_prob=1.0,
        cutout_area_range=(area, area),
        cutout_one_large_prob=1.0,    # always one big rectangle
        max_shift=0, resolution_factor=1.0,
        gaussian_blur_sigma=0.0, height_noise_sigma=0.0,
        full_dropout_prob=0.0, always_on_noise_sigma=0.0,
    )


REGIMES = {
    "clean":         None,
    "shift_24":      _make_shift_corruptor(24),
    "shift_48":      _make_shift_corruptor(48),
    "cutout_25pct":  _make_cutout_corruptor(0.25),
    "cutout_50pct":  _make_cutout_corruptor(0.50),
}


def _seed_for(frame_id: str, regime: str) -> int:
    return abs(hash((frame_id, regime))) % (2**31 - 1)


def _apply_regime(chm_clean: np.ndarray, regime: str, frame_id: str) -> np.ndarray:
    if regime == "clean":
        return chm_clean.copy()
    seed = _seed_for(frame_id, regime)
    random.seed(seed)
    np.random.seed(seed)
    return REGIMES[regime](chm_clean.copy())


# --------------------------------------------------------------------------
# Hp7 wrapper (per-sample minmax norm + sigmoid denorm)
# --------------------------------------------------------------------------

class Hp7Wrapper:
    def __init__(self, ckpt_path: Path, device: torch.device,
                 minmax_min_scale: float = 0.5):
        self.device = device
        self.minmax_min_scale = minmax_min_scale
        log.info("Hp7: loading %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["hyper_parameters"]["config"]
        net = get_model(cfg)
        module = HeightEstimationModule(model=net, config=cfg)
        module.load_state_dict(ckpt["state_dict"], strict=True)
        module.eval().to(device)
        self.module = module

    def _per_sample_minmax(self, chm: np.ndarray):
        valid = (chm > 0) & np.isfinite(chm)
        if not valid.any():
            return chm.astype(np.float32), 0.0, 1.0
        h_min = float(chm[valid].min())
        h_max = float(chm[valid].max())
        scale = max(h_max - h_min, float(self.minmax_min_scale))
        return ((chm - h_min) / scale).astype(np.float32), h_min, scale

    def predict(self, image_u8: np.ndarray, chm_metric_clean: np.ndarray,
                chm_metric_corrupted: np.ndarray) -> np.ndarray:
        """Return depth in metres.

        ``chm_metric_clean`` derives (shift, scale) — matches the
        PromptDA-style "deployable LiDAR is the shift/scale anchor"
        convention. ``chm_metric_corrupted`` is what enters the network,
        normalised with the same (shift, scale).
        """
        _, shift, scale = self._per_sample_minmax(chm_metric_clean)
        chm_norm = ((chm_metric_corrupted - shift) / scale).astype(np.float32)

        img = (image_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=torch.float32,
        )
        chm_t = (
            torch.from_numpy(chm_norm)
            .unsqueeze(0).unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True):
            pred_norm = self.module(img_t, chm_t)
        pred_norm = pred_norm.float().squeeze().cpu().numpy()
        return pred_norm * scale + shift


# --------------------------------------------------------------------------
# PromptDA wrapper (uses upstream model's internal normalisation)
# --------------------------------------------------------------------------

class PromptDAWrapper:
    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        from promptda.promptda import PromptDA  # noqa: E402
        log.info("PromptDA: loading %s", model_id)
        self.model = PromptDA.from_pretrained(model_id).to(device).eval()

    def predict(self, image_u8: np.ndarray, _chm_clean: np.ndarray,
                chm_metric_corrupted: np.ndarray) -> np.ndarray:
        img_t = (
            torch.from_numpy(image_u8.astype(np.float32) / 255.0)
            .permute(2, 0, 1).unsqueeze(0)
            .to(self.device)
        )
        chm_t = (
            torch.from_numpy(chm_metric_corrupted.astype(np.float32))
            .unsqueeze(0).unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            pred = self.model.predict(img_t, chm_t)
        return pred.squeeze().float().cpu().numpy()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mask = gt > 0
    if not mask.any():
        return dict(mae_m=float("nan"), rmse_m=float("nan"), delta1=float("nan"),
                    n_valid=0)
    p = pred[mask]
    g = gt[mask]
    err = np.abs(p - g)
    ratio = np.maximum(p / np.maximum(g, 1e-6), g / np.maximum(p, 1e-6))
    return dict(
        mae_m=float(err.mean()),
        rmse_m=float(np.sqrt((err ** 2).mean())),
        delta1=float((ratio < 1.25).mean()),
        n_valid=int(mask.sum()),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_hp7 = PACKAGE_ROOT / "weights" / "Hp7_hypersim_last.ckpt"
    default_data = PACKAGE_ROOT / "data" / "arkitscenes" / "upsampling"
    default_results = (PACKAGE_ROOT / "benchmarking" / "results" / "arkitscenes")

    parser.add_argument("--hp7_ckpt", type=Path, default=default_hp7,
                        help=f"Hp7 .ckpt (default: {default_hp7.relative_to(PACKAGE_ROOT)})")
    parser.add_argument("--promptda_id",
                        default="depth-anything/prompt-depth-anything-vitl",
                        help="HF id of the PromptDA baseline")
    parser.add_argument("--data_dir", type=Path, default=default_data,
                        help="ARKitScenes upsampling root (parent of "
                             "Validation/). Default assumes "
                             "scripts/download_data.sh --arkitscenes was "
                             "run from this package.")
    parser.add_argument("--max_total_frames", type=int, default=2000,
                        help="Cap on total frames evaluated (round-robin "
                             "across all videos). Set to 0 for the full set "
                             "(~17k frames at the full default sampling).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_json", type=Path,
                        default=default_results / "results_hp7_vs_promptda_arkit_full.json",
                        help="Per-frame rows.")
    parser.add_argument("--out_summary", type=Path,
                        default=default_results / "results_hp7_vs_promptda_arkit_summary.json",
                        help="Aggregated mean/median/p95 MAE per regime.")
    parser.add_argument("--out_markdown", type=Path,
                        default=default_results / "results_hp7_vs_promptda_arkit_summary.md",
                        help="Markdown summary table.")
    parser.add_argument("--checkpoint_every", type=int, default=200,
                        help="Flush partial JSON / summary every N frames "
                             "so a crash mid-run doesn't lose everything.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA required (no CPU fallback). "
                 "See README.md hardware requirements.")

    val_dir = args.data_dir / "Validation"
    if not val_dir.is_dir():
        sys.exit(f"ARKitScenes validation directory not found: {val_dir}\n"
                 f"Run 'bash scripts/download_data.sh --arkitscenes' first.")

    cap = args.max_total_frames if args.max_total_frames > 0 else None
    triplets = _list_all_validation_frames(val_dir, max_total=cap)
    log.info("Will evaluate %d frames across %d videos",
             len(triplets), len({t[0] for t in triplets}))

    device = torch.device(args.device)
    hp7 = Hp7Wrapper(args.hp7_ckpt, device)
    promptda = PromptDAWrapper(args.promptda_id, device)

    rows: list[dict] = []
    summary = {m: {r: [] for r in REGIMES} for m in ("hp7", "promptda")}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)

    def _write_summary(progress_label: str = "final"):
        agg: dict = {"label": progress_label}
        for model_name in ("hp7", "promptda"):
            agg[model_name] = {}
            base = float(np.mean(summary[model_name]["clean"])) \
                if summary[model_name]["clean"] else float("nan")
            for regime in REGIMES:
                vals = summary[model_name][regime]
                if not vals:
                    continue
                agg[model_name][regime] = {
                    "n": len(vals),
                    "mae_mean_m": round(float(np.mean(vals)), 4),
                    "mae_median_m": round(float(np.median(vals)), 4),
                    "mae_p95_m": round(float(np.percentile(vals, 95)), 4),
                    "delta_clean_mean_m": round(float(np.mean(vals)) - base, 4),
                }
        args.out_summary.write_text(json.dumps(agg, indent=2))

    def _flush_results(progress_label: str):
        args.out_json.write_text(json.dumps(rows, indent=2))
        _write_summary(progress_label)

    t_start = time.time()
    for idx, (video_id, color_path, lowres_path, highres_path) in enumerate(triplets):
        frame_id = f"{video_id}/{color_path.stem}"
        try:
            image_u8, chm_clean, gt = _load_frame(color_path, lowres_path, highres_path)
        except Exception as e:
            log.warning("[%d/%d] %s — skipped (%s)", idx + 1, len(triplets),
                        frame_id, e)
            continue

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1e-6)
            eta_s = (len(triplets) - (idx + 1)) / max(rate, 1e-6)
            log.info("[%d/%d] %s  |  %.2f frame/s  |  ETA %.1f min",
                     idx + 1, len(triplets), frame_id,
                     rate, eta_s / 60)

        for regime in REGIMES:
            chm_corr = _apply_regime(chm_clean, regime, frame_id)
            for model_name, runner in [("hp7", hp7), ("promptda", promptda)]:
                try:
                    pred = runner.predict(image_u8, chm_clean, chm_corr)
                except Exception as e:
                    log.warning("  %s on %s/%s failed (%s)",
                                model_name, frame_id, regime, e)
                    continue
                m = _metrics(pred, gt)
                row = {
                    "frame": frame_id, "regime": regime, "model": model_name,
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in m.items()},
                }
                rows.append(row)
                if not np.isnan(m["mae_m"]):
                    summary[model_name][regime].append(m["mae_m"])

        if args.checkpoint_every and (idx + 1) % args.checkpoint_every == 0:
            _flush_results(progress_label=f"partial_{idx + 1}")
            log.info("  -> partial flush: %d rows so far", len(rows))

    total_s = time.time() - t_start
    log.info("Total runtime: %.1f s for %d frames × %d regimes × 2 models",
             total_s, len(triplets), len(REGIMES))

    print("")
    print("=" * 88)
    print(f"Hp7 ckpt: {args.hp7_ckpt.name}")
    print(f"PromptDA: {args.promptda_id}")
    n_videos = len({t[0] for t in triplets})
    print(f"Frames: {len(triplets)}  (across {n_videos} videos)")
    print("=" * 88)
    print(f"{'regime':<14} | {'Hp7 MAE [m]':>30} | {'PromptDA MAE [m]':>30}")
    print(f"{'':<14} | {'mean':>8} {'med':>8} {'Δclean':>10} | "
          f"{'mean':>8} {'med':>8} {'Δclean':>10}")
    print("-" * 88)
    base_hp7 = float(np.mean(summary["hp7"]["clean"])) if summary["hp7"]["clean"] else float("nan")
    base_pda = float(np.mean(summary["promptda"]["clean"])) if summary["promptda"]["clean"] else float("nan")
    md_lines = [
        "# Hp7 vs PromptDA on ARKitScenes Validation",
        "",
        f"- **Hp7 ckpt**: `{args.hp7_ckpt.name}`",
        f"- **PromptDA**: `{args.promptda_id}`",
        f"- **Frames**: {len(triplets)} across {n_videos} videos",
        f"- **Runtime**: {total_s:.1f} s ({len(triplets) * len(REGIMES) * 2} forward passes)",
        "",
        "| Regime | Hp7 mean | Hp7 med | Hp7 Δclean | PromptDA mean | PromptDA med | PromptDA Δclean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        hp7_vals = summary["hp7"][regime]
        pda_vals = summary["promptda"][regime]
        hp7_mean = float(np.mean(hp7_vals)) if hp7_vals else float("nan")
        hp7_med = float(np.median(hp7_vals)) if hp7_vals else float("nan")
        pda_mean = float(np.mean(pda_vals)) if pda_vals else float("nan")
        pda_med = float(np.median(pda_vals)) if pda_vals else float("nan")
        hp7_delta = hp7_mean - base_hp7
        pda_delta = pda_mean - base_pda
        print(f"{regime:<14} | "
              f"{hp7_mean:>8.3f} {hp7_med:>8.3f} {hp7_delta:>+10.3f} | "
              f"{pda_mean:>8.3f} {pda_med:>8.3f} {pda_delta:>+10.3f}")
        md_lines.append(
            f"| `{regime}` | {hp7_mean:.3f} | {hp7_med:.3f} | {hp7_delta:+.3f} | "
            f"{pda_mean:.3f} | {pda_med:.3f} | {pda_delta:+.3f} |"
        )
    md_lines.append("")
    md_lines.append("All MAE values in metres, computed over `gt > 0` pixels per frame.")

    _flush_results("final")
    args.out_markdown.write_text("\n".join(md_lines))
    print("")
    log.info("Per-frame rows  → %s (%d rows)", args.out_json, len(rows))
    log.info("Aggregate JSON  → %s", args.out_summary)
    log.info("Markdown table  → %s", args.out_markdown)


if __name__ == "__main__":
    main()
