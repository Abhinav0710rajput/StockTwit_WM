"""
ELBO loss for the Twit Wave RSSM.

L = L_BCE  +  λ * L_MSE  +  β * max(free_nats, KL)

L_BCE : weighted binary cross-entropy over all vocab tickers (presence)
L_MSE : MSE over the 5 features for active tickers only
KL    : analytical KL between posterior and prior (diagonal Gaussians)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.rssm import kl_divergence


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
    free_nats: float = 3.0,
    pos_weight: float = 10.0,        # upweight positives to handle class imbalance
) -> dict[str, torch.Tensor]:
    """
    Returns dict with keys: total, bce, mse, kl  (all scalars).
    """
    B, T, V = presence_logits.shape

    # --- BCE loss (presence) ---
    pw = torch.tensor(pos_weight, device=presence_logits.device)
    bce = F.binary_cross_entropy_with_logits(
        presence_logits.reshape(B * T, V),
        presence_targets.reshape(B * T, V),
        pos_weight=pw,
        reduction="mean",
    )

    # --- MSE loss (features, active tickers only) ---
    active_mask = (ticker_ids != 0).float()          # (B, T, N)
    n_active = active_mask.sum().clamp(min=1)
    sq_err = ((feat_pred - feat_true) ** 2).mean(dim=-1)  # (B, T, N)
    mse = (sq_err * active_mask).sum() / n_active

    # --- KL loss ---
    # Reshape to (B*T, s_dim) for reuse of kl_divergence
    BT = B * T
    kl_per_step = kl_divergence(
        post_mean.reshape(BT, -1),
        post_logvar.reshape(BT, -1),
        prior_mean.reshape(BT, -1),
        prior_logvar.reshape(BT, -1),
    )   # (B*T,)
    kl = kl_per_step.mean()
    kl_clipped = torch.clamp(kl, min=free_nats)

    total = bce + lambda_ * mse + beta * kl_clipped

    return {
        "total": total,
        "bce":   bce,
        "mse":   mse,
        "kl":    kl,
        "kl_clipped": kl_clipped,
    }
