"""GreenEarthNet Vegetation Score and a differentiable NNSE proxy."""
import torch


def per_pixel_nse(pred: torch.Tensor, targ: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    Per-pixel Nash-Sutcliffe efficiency over the time axis, restricted to ``mask``.

    Args:
        pred, targ: (B, T, H, W)
        mask:       (B, T, H, W) float — 1 where the observation is valid (clear sky)
    Returns:
        nse:   (B, H, W) — NaN for pixels with <2 valid obs or zero variance
        valid: (B, H, W) bool — pixels where NSE is defined
    """
    n = mask.sum(dim=1)
    has_obs = n >= 2

    targ_mean = (targ * mask).sum(dim=1) / n.clamp(min=1)
    num = ((pred - targ) ** 2 * mask).sum(dim=1)
    den = ((targ - targ_mean.unsqueeze(1)) ** 2 * mask).sum(dim=1)

    valid = has_obs & (den > eps)
    nse = torch.where(valid, 1.0 - num / den.clamp(min=eps), torch.full_like(num, float("nan")))
    return nse, valid


def nnse_per_pixel(pred: torch.Tensor, targ: torch.Tensor, mask: torch.Tensor):
    """Normalized NSE in [0, 1]; NaN where undefined."""
    nse, valid = per_pixel_nse(pred, targ, mask)
    nnse = 1.0 / (2.0 - nse)
    return nnse, valid


def vegetation_score(
    pred_ndvi: torch.Tensor,
    targ_ndvi: torch.Tensor,
    cloud_mask_future: torch.Tensor,
    veg_mask: torch.Tensor,
) -> torch.Tensor:
    """
    GreenEarthNet Vegetation Score.

    Args:
        pred_ndvi, targ_ndvi: (B, T, H, W) — NDVI in raw space
        cloud_mask_future:    (B, T, H, W) bool — True = clear sky
        veg_mask:             (B, H, W) bool — True = natural vegetation
    """
    targ_ndvi = torch.nan_to_num(targ_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    pred_ndvi = torch.nan_to_num(pred_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    obs_mask = (cloud_mask_future & veg_mask.unsqueeze(1)).float()
    nnse, valid = nnse_per_pixel(pred_ndvi, targ_ndvi, obs_mask)
    pix_mask = valid & veg_mask
    if pix_mask.sum() == 0:
        return torch.tensor(float("nan"), device=pred_ndvi.device)
    mean_nnse = nnse[pix_mask].mean()
    return 2.0 - 1.0 / mean_nnse


def nnse_loss(
    pred_ndvi: torch.Tensor,
    targ_ndvi: torch.Tensor,
    cloud_mask_future: torch.Tensor,
    veg_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable proxy: 1 - mean(NNSE) over vegetation pixels.

    Uses the same masking as ``vegetation_score`` so train and eval are aligned.
    Returns 0 if no valid pixels (keeps the optimizer stable on degenerate batches).
    """
    # NaN in masked-out (fully-clouded) pixels would poison the backward graph
    # via 0 * NaN = NaN — replace with 0 since they're gated out anyway.
    targ_ndvi = torch.nan_to_num(targ_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    pred_ndvi = torch.nan_to_num(pred_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    obs_mask = (cloud_mask_future & veg_mask.unsqueeze(1)).float()
    n = obs_mask.sum(dim=1)
    has_obs = n >= 2

    targ_mean = (targ_ndvi * obs_mask).sum(dim=1) / n.clamp(min=1)
    num = ((pred_ndvi - targ_ndvi) ** 2 * obs_mask).sum(dim=1)
    den = ((targ_ndvi - targ_mean.unsqueeze(1)) ** 2 * obs_mask).sum(dim=1)

    valid = has_obs & (den > 1e-6) & veg_mask
    if valid.sum() == 0:
        return pred_ndvi.sum() * 0.0
    nse = 1.0 - num / den.clamp(min=1e-6)
    nnse = 1.0 / (2.0 - nse.clamp(max=1.0 - 1e-4))
    return 1.0 - nnse[valid].mean()


def masked_ndvi_mse(
    pred_ndvi: torch.Tensor,
    targ_ndvi: torch.Tensor,
    cloud_mask_future: torch.Tensor,
    veg_mask: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
    huber_delta: float | None = None,
) -> torch.Tensor:
    """MSE (or Huber if ``huber_delta`` is set) on raw NDVI, restricted to clear vegetation pixels."""
    targ_ndvi = torch.nan_to_num(targ_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    pred_ndvi = torch.nan_to_num(pred_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (cloud_mask_future & veg_mask.unsqueeze(1)).float()  # (B,T,H,W)
    if horizon_weights is not None:
        w = horizon_weights.to(pred_ndvi.dtype).view(1, -1, 1, 1)
        mask = mask * w
    diff = pred_ndvi - targ_ndvi
    if huber_delta is None:
        sq = diff * diff
    else:
        abs_diff = diff.abs()
        quad = 0.5 * diff * diff
        lin = huber_delta * (abs_diff - 0.5 * huber_delta)
        sq = torch.where(abs_diff <= huber_delta, quad, lin)
    sq = sq * mask
    return sq.sum() / mask.sum().clamp(min=1.0)


def masked_delta_mse(
    pred_delta: torch.Tensor,
    targ_delta: torch.Tensor,
    cloud_mask_future: torch.Tensor,
    veg_mask: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
    huber_delta: float | None = None,
) -> torch.Tensor:
    """MSE (or Huber if ``huber_delta`` is set) on tanh-normalized S2 deltas, restricted to clear vegetation pixels.

    If ``horizon_weights`` is given (shape ``(T,)``), per-step errors are
    reweighted — the weights are folded into both numerator and denominator so
    the loss magnitude stays comparable to the unweighted version.
    """
    targ_delta = torch.nan_to_num(targ_delta, nan=0.0, posinf=0.0, neginf=0.0)
    pred_delta = torch.nan_to_num(pred_delta, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (cloud_mask_future & veg_mask.unsqueeze(1)).float().unsqueeze(2)  # (B,T,1,H,W)
    if horizon_weights is not None:
        w = horizon_weights.to(pred_delta.dtype).view(1, -1, 1, 1, 1)
        mask = mask * w
    diff = pred_delta - targ_delta
    if huber_delta is None:
        sq = diff * diff
    else:
        abs_diff = diff.abs()
        quad = 0.5 * diff * diff
        lin = huber_delta * (abs_diff - 0.5 * huber_delta)
        sq = torch.where(abs_diff <= huber_delta, quad, lin)
    sq = sq * mask
    denom = mask.sum().clamp(min=1.0) * pred_delta.shape[2]
    return sq.sum() / denom
