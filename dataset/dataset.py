import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.loader import load_tile

WEATHER_VARS = ["eobs_hu", "eobs_rr", "eobs_pp", "eobs_qq", "eobs_tg", "eobs_tn", "eobs_tx"]
S2_VARS = ["s2_B02", "s2_B03", "s2_B04", "s2_B8A"]
DEM_VARS = ["nasa_dem", "cop_dem", "alos_dem"]

# ESA WorldCover classes considered vegetation
VEG_CLASSES = {10, 20, 30}  # Trees, Shrubland, Grassland


def _tanh_normalize(x: np.ndarray, mean: float, std: float, c: float = 3.0) -> np.ndarray:
    z = (x - mean) / (std + 1e-8)
    return np.tanh(z / c)


def _coarsen_weather(weather_ds) -> np.ndarray:
    """Coarsen daily weather (150 steps) to 5-daily (30 steps). Returns (30, 7)."""
    arrays = []
    for v in WEATHER_VARS:
        da = weather_ds[v]
        if v == "eobs_rr":
            coarsened = da.coarsen(time=5).sum().values.astype(np.float32)
        else:
            coarsened = da.coarsen(time=5).mean().values.astype(np.float32)
        arrays.append(coarsened)
    return np.stack(arrays, axis=1)  # (30, 7)


def _compute_baseline(s2: np.ndarray, cloudmask: np.ndarray) -> np.ndarray:
    """
    Compute cloud-free baseline X_base from the context period (t=0..9).

    Walk forward in time; the last clear observation per pixel is the baseline.
    Pixels with no clear observation fall back to the temporal mean, or global mean.

    Args:
        s2:        (T=30, C=4, H, W) — scaled S2 reflectance
        cloudmask: (T=30, H, W) — 0 = clear sky

    Returns:
        baseline: (C=4, H, W)
    """
    ctx_s2 = s2[:10]          # (10, 4, H, W)
    ctx_cloud = cloudmask[:10]  # (10, H, W)
    clear = ctx_cloud == 0     # (10, H, W)

    # Fallback: mean of clear pixels; if the pixel has no clear context obs,
    # fall back to the spatial mean of clear pixels in this tile (no cloudy leak).
    clear_4d = clear[:, np.newaxis]  # (10, 1, H, W)
    count = clear_4d.sum(axis=0)     # (1, H, W)
    ctx_s2_safe = np.nan_to_num(ctx_s2, nan=0.0)
    clear_sum_time = (ctx_s2_safe * clear_4d).sum(axis=0)            # (4, H, W)
    clear_sum_global = clear_sum_time.sum(axis=(1, 2), keepdims=True)  # (4, 1, 1)
    clear_count_global = clear_4d.sum(axis=(0, 2, 3), keepdims=True)[0]  # (1, 1, 1)
    global_mean = clear_sum_global / np.maximum(clear_count_global, 1)
    mean_clear = np.where(
        count > 0,
        clear_sum_time / np.maximum(count, 1),
        np.broadcast_to(global_mean, ctx_s2.shape[1:]),
    )  # (4, H, W)

    # Forward-pass: keep updating with each clear observation → last clear wins
    baseline = mean_clear.copy()
    for t in range(10):
        mask = clear[t][np.newaxis]  # (1, H, W)
        baseline = np.where(mask, ctx_s2[t], baseline)

    # Guard: if all context pixels were NaN/cloudy the baseline can still be NaN.
    # Replace residual NaNs with 0 (neutral for tanh-normalized deltas).
    baseline = np.nan_to_num(baseline, nan=0.0)
    return baseline  # (4, H, W)


def _impute_context(s2_ctx: np.ndarray, cloudmask_ctx: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Fill cloudy context pixels with the cloud-free baseline.

    Args:
        s2_ctx:       (10, 4, H, W)
        cloudmask_ctx:(10, H, W)
        baseline:     (4, H, W)

    Returns:
        imputed: (10, 4, H, W)
    """
    clear = (cloudmask_ctx == 0)[:, np.newaxis]  # (10, 1, H, W)
    return np.where(clear, s2_ctx, baseline[np.newaxis])


class GreenEarthNetDataset(Dataset):
    """
    PyTorch Dataset for GreenEarthNet minicubes.

    Returns a dict with:
        context:            (10, 16, 128, 128) float32 — tanh-normalized
        future:             (20, 10, 128, 128) float32 — tanh-normalized
        target_delta:       (20,  4, 128, 128) float32 — tanh-normalized deltas
        baseline_s2:        ( 4, 128, 128) float32 — raw cloud-free baseline reflectance
        target_ndvi:        (20, 128, 128) float32 — raw NDVI for forecast period
        veg_mask:           (128, 128) bool — True for vegetation pixels
        cloud_mask_future:  (20, 128, 128) bool — True for clear-sky pixels (t=10..29)
        doy_future:         (20,) float32 — day-of-year for forecast steps
    """

    def __init__(self, root_dir: str | Path, stats_path: str | Path):
        self.files = sorted(Path(root_dir).rglob("*.nc"))
        if len(self.files) == 0:
            raise ValueError(f"No .nc files found under {root_dir}")

        with open(stats_path) as f:
            self.stats = json.load(f)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        tile_data = load_tile(self.files[idx])

        # --- Weather: coarsen 150 daily → 30 5-daily, shape (30, 7) ---
        weather_np = _coarsen_weather(tile_data["weather"])

        # --- S2: convert to float32, shape (30, 4, 128, 128) ---
        s2_np = np.stack(
            [tile_data["s2"][v].values.astype(np.float32) for v in S2_VARS], axis=1
        )  # (30, 4, 128, 128)

        # --- Cloudmask: shape (30, 128, 128), 0 = clear; NaN → treat as cloudy (1) ---
        mask_var = list(tile_data["cloudmask"].data_vars)[0]
        _cm_raw = tile_data["cloudmask"][mask_var].values
        cloudmask_np = np.where(np.isfinite(_cm_raw), np.round(_cm_raw), 1).astype(np.int32)

        # --- DEM: shape (3, 128, 128) ---
        dem_np = np.stack(
            [tile_data["dem"][v].values.astype(np.float32) for v in DEM_VARS], axis=0
        )  # (3, 128, 128)

        # --- LULC: shape (128, 128) ---
        lulc_np = tile_data["lulc"]["esawc_lc"].values  # (128, 128)

        # --- Cloud-free baseline from context (t=0..9); used as NaN fill + evaluator persistence ---
        baseline = _compute_baseline(s2_np, cloudmask_np)  # (4, 128, 128)

        # --- Fill NaN anywhere in s2 with the baseline BEFORE imputation, so clear-but-NaN
        #     pixels don't leak NaN into s2_ctx_imputed (and thus into s2_anchor). ---
        s2_np = np.where(np.isfinite(s2_np), s2_np, baseline[np.newaxis])

        # --- Impute cloudy context pixels with baseline ---
        s2_ctx_imputed = _impute_context(s2_np[:10], cloudmask_np[:10], baseline)  # (10, 4, H, W)
        s2_np[:10] = s2_ctx_imputed

        # --- Raw target NDVI for forecast period (metric space) ---
        fut_b04 = s2_np[10:, 2]
        fut_b8a = s2_np[10:, 3]
        target_ndvi = (fut_b8a - fut_b04) / (fut_b8a + fut_b04 + 1e-6)  # (20, H, W)

        # --- Anchor = s2 at last context step (t=9), cloud-imputed ---
        s2_anchor = s2_ctx_imputed[-1]  # (4, 128, 128)

        # --- Absolute offsets from anchor: delta[t] = s2[t] - s2_anchor ---
        target_delta = (s2_np[10:] - s2_anchor[np.newaxis]).astype(np.float32)  # (20, 4, H, W)

        # --- Delta loss mask: clear at t (anchor is already imputed) ---
        cloud_mask_delta = cloudmask_np[10:] == 0                               # (20, H, W)

        # --- Normalize weather ---
        weather_norm = np.stack(
            [_tanh_normalize(weather_np[:, i], **self.stats[v])
             for i, v in enumerate(WEATHER_VARS)],
            axis=1,
        )  # (30, 7)

        # --- Normalize DEM ---
        dem_norm = np.stack(
            [_tanh_normalize(dem_np[i], **self.stats[v])
             for i, v in enumerate(DEM_VARS)],
            axis=0,
        )  # (3, 128, 128)

        # --- Broadcast DEM to spatial dims (weather stays a (30,7) timeseries,
        #     injected into the model via FiLM / encoder projection). ---
        H, W = 128, 128
        dem_tiled = np.tile(dem_norm[np.newaxis], (30, 1, 1, 1))         # (30, 3, H, W)

        # --- Cloudmask channel (float, 0=clear, 1=cloud) for context ---
        cloudmask_ctx_chan = (cloudmask_np[:10] != 0).astype(np.float32)[:, np.newaxis]  # (10, 1, H, W)

        # --- NDVI proxy channel from raw reflectance ---
        ctx_b04 = s2_ctx_imputed[:, 2]
        ctx_b8a = s2_ctx_imputed[:, 3]
        ndvi_ctx = ((ctx_b8a - ctx_b04) / (ctx_b8a + ctx_b04 + 1e-6))[:, np.newaxis]  # (10, 1, H, W)

        # --- Assemble context tensor: [s2_raw(4)|mask(1)|dem(3)|ndvi(1)] ---
        context = np.concatenate(
            [s2_ctx_imputed, cloudmask_ctx_chan, dem_tiled[:10], ndvi_ctx], axis=1
        )  # (10, 9, H, W)

        # --- Future decoder input uses raw baseline reflectance + its NDVI ---
        baseline_ndvi = (baseline[3] - baseline[2]) / (baseline[3] + baseline[2] + 1e-6)  # (H, W)
        baseline_spatial = np.concatenate(
            [baseline, baseline_ndvi[np.newaxis]], axis=0
        )  # (5, H, W)
        baseline_tiled = np.tile(baseline_spatial[np.newaxis], (20, 1, 1, 1))  # (20, 5, H, W)

        # --- Assemble future tensor: [baseline_s2(4)|baseline_ndvi(1)|dem(3)] ---
        future = np.concatenate([baseline_tiled, dem_tiled[10:]], axis=1)  # (20, 8, H, W)

        # --- Vegetation mask ---
        veg_mask = np.isin(lulc_np, list(VEG_CLASSES))  # (H, W) bool

        # --- Cloud mask for forecast period: True = clear sky (used by NDVI metric) ---
        cloud_mask_future = cloudmask_np[10:] == 0  # (20, H, W) bool

        # Final NaN/Inf guard.
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        future = np.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)
        target_delta = np.nan_to_num(target_delta, nan=0.0, posinf=0.0, neginf=0.0)
        weather_norm = np.nan_to_num(weather_norm, nan=0.0, posinf=0.0, neginf=0.0)

        doy_all = np.asarray(tile_data["doy"], dtype=np.float32)  # (30,)
        doy_context = torch.from_numpy(doy_all[:10])  # (10,)
        doy_future = torch.from_numpy(doy_all[10:])   # (20,)

        weather_context = torch.from_numpy(weather_norm[:10].astype(np.float32))  # (10, 7)
        weather_future = torch.from_numpy(weather_norm[10:].astype(np.float32))   # (20, 7)

        return {
            "context": torch.from_numpy(context.astype(np.float32)),
            "future": torch.from_numpy(future.astype(np.float32)),
            "target_delta": torch.from_numpy(target_delta),
            "s2_anchor": torch.from_numpy(s2_anchor.astype(np.float32)),
            "baseline_s2": torch.from_numpy(baseline.astype(np.float32)),
            "target_ndvi": torch.from_numpy(target_ndvi.astype(np.float32)),
            "veg_mask": torch.from_numpy(veg_mask),
            "cloud_mask_future": torch.from_numpy(cloud_mask_future),
            "cloud_mask_delta": torch.from_numpy(cloud_mask_delta),
            "doy_context": doy_context,
            "doy_future": doy_future,
            "weather_context": weather_context,
            "weather_future": weather_future,
            "lulc": torch.from_numpy(lulc_np.astype(np.int16)),
        }


if __name__ == "__main__":
    import sys

    stats_path = Path("stats.json")
    if not stats_path.exists():
        print("stats.json not found — run compute_stats.py first")
        sys.exit(1)

    ds = GreenEarthNetDataset("greenearthnet/train", stats_path)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
