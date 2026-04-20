"""Lightweight dataset that loads pre-computed .pt sample files."""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """
    Reads sample dicts produced by ``precompute_cache.py``.

    Float tensors are stored as bfloat16 on disk (~2x smaller than fp32, full fp32 range)
    and cast back to fp32 on load. Bool masks and doy_future stay in their native dtype.
    """

    FLOAT_KEYS = {
        "context", "future", "target_delta", "baseline_s2", "target_ndvi",
        "weather_context", "weather_future",
    }

    def __init__(self, cache_dir: str | Path):
        self.files = sorted(Path(cache_dir).rglob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt files found under {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        sample = torch.load(self.files[idx], weights_only=True, map_location="cpu")
        out = {}
        for k, v in sample.items():
            if k in self.FLOAT_KEYS:
                if v.dtype != torch.float32:
                    v = v.to(torch.float32)
                # Defensive NaN/Inf scrub — protects training from any stale
                # or partially-rebuilt caches. Cheap (one pass per tensor).
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            out[k] = v
        return out


class AugmentedCachedDataset(CachedDataset):
    """Random D4 (rot90 × flip = 8 orientations) on the spatial dims of every spatial tensor."""

    SPATIAL_KEYS = {
        "context", "future", "target_delta", "target_ndvi",
        "baseline_s2", "s2_anchor", "veg_mask",
        "cloud_mask_future", "cloud_mask_delta", "lulc",
    }

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        k = random.randint(0, 3)
        flip = random.random() < 0.5
        for key in self.SPATIAL_KEYS:
            if key not in sample:
                continue
            v = sample[key]
            if k:
                v = torch.rot90(v, k, dims=(-2, -1))
            if flip:
                v = torch.flip(v, dims=(-1,))
            sample[key] = v.contiguous()
        return sample
