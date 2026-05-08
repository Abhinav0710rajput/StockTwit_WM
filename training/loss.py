"""
ELBO loss for the Twit Wave RSSM.

L = L_BCE  +  λ * L_MSE  +  β * KL_fb

KL_fb uses per-latent-dim free bits: KL_fb = mean_dim( max(free_nats, kl_d) )
This gives each dim a free information floor before any KL penalty kicks in,
matching PlaNet / Dreamer convention. KL is averaged (not summed) over dims so
its scale matches the per-element BCE, making β directly interpretable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def elbo_loss(
    presence_logits: torch.Tensor,   # (B, T, vocab_size)
    presence_targets: torch.Tensor,  # (B, T, vocab_size)  float {0,1}
    feat_pred: torch.Tensor,         # (B, T, N, 5)
    feat_true: torch.Tensor,         # (B, T, N, 5)
    ticker_ids: torch.Tensor,        # (B, T, N)  — 0 = padding
    post_mean: torch.Tensor,         # (B, T, s_dim)
    post_logvar: torch.Tensor,
    prior_mean: torch.Tensor,        # (B, T, s_dim)
    prior_logvar: torch.Tensor,
    lambda_: float = 1.0,
    beta: float = 1.0,
    free_nats: float = 1.0,
    pos_weight: float = 10.0,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Returns dict with keys: total, bce, mse, kl, kl_clipped  (all scalars).
    """
    B, T, V = presence_logits.shape

    # --- BCE loss (presence) ---
    pw = torch.tensor(pos_weight, device=presence_logits.device)
    targets = presence_targets.reshape(B * T, V)
    if label_smoothing > 0.0:
        # push positives toward (1 - ε/2), negatives toward ε/2
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(
        presence_logits.reshape(B * T, V),
        targets,
        pos_weight=pw,
        reduction="mean",
    )

    # --- MSE loss (features, active tickers only) ---
    active_mask = (ticker_ids != 0).float()
    n_active = active_mask.sum().clamp(min=1)
    sq_err = ((feat_pred - feat_true) ** 2).mean(dim=-1)   # (B, T, N)
    mse = (sq_err * active_mask).sum() / n_active

    # --- KL loss with per-dim free bits ---
    # Compute per-dim KL: (B*T, s_dim) — do NOT sum over dims yet.
    # Averaging over dims keeps KL on the same per-element scale as BCE,
    # so β is directly comparable across the two terms.
    BT = B * T
    pm  = post_mean.reshape(BT, -1)
    plv = post_logvar.reshape(BT, -1)
    rm  = prior_mean.reshape(BT, -1)
    rlv = prior_logvar.reshape(BT, -1)

    prior_var = rlv.exp()
    post_var  = plv.exp()
    kl_per_dim = 0.5 * (
        rlv - plv
        + (post_var + (pm - rm) ** 2) / prior_var.clamp(min=1e-6)
        - 1.0
    )   # (B*T, s_dim)

    # Free bits: floor each dim independently, then mean over dims and B*T
    kl_clipped = kl_per_dim.clamp(min=free_nats).mean()
    kl = kl_per_dim.mean()   # unclipped, for logging

    total = bce + lambda_ * mse + beta * kl_clipped

    return {
        "total":      total,
        "bce":        bce,
        "mse":        mse,
        "kl":         kl,
        "kl_clipped": kl_clipped,
    }
