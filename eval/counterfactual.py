"""
Counterfactual probing: perturb the latent representation of one ticker
and measure how other tickers' predicted features change.

Tests the ecosystem / finite-attention hypothesis:
  - Spiking a meme stock should crowd out unrelated tickers
  - Related tickers should rise (sentiment contagion)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from model.twit_wave import TwitWave
from data.vocab import Vocabulary


def _get_ticker_contribution(
    model: TwitWave,
    z: torch.Tensor,          # (1, z_dim)
    target_id: int,
    all_ids: torch.Tensor,    # (1, N) — all active tickers
    device: torch.device,
) -> torch.Tensor:
    """Decode features for all_ids given z. Returns (N, 5)."""
    return model.decode_features(z, all_ids.to(device))[0]


def run_counterfactual(
    model: TwitWave,
    vocab: Vocabulary,
    features_seq: torch.Tensor,    # (T_ctx, N, 5)
    ticker_ids_seq: torch.Tensor,  # (T_ctx, N)
    target_ticker: str,
    delta_log_attn: float,         # perturbation magnitude on log_attn feature
    eval_tickers: list[str],       # which tickers to track in output
    device: torch.device,
    window_k: int | None = None,
) -> pd.DataFrame:
    """
    1. Run context phase → z_T
    2. Decode original features for eval_tickers
    3. Perturb z_T in the direction that increases target ticker's log_attn
    4. Decode perturbed features for eval_tickers
    5. Report Δ features

    Returns DataFrame with columns: ticker, feat, original, perturbed, delta
    """
    if window_k is None:
        window_k = model.cfg.window_k

    model.eval()
    target_id = vocab.encode(target_ticker)
    eval_ids  = torch.tensor(
        [vocab.encode(t) for t in eval_tickers], dtype=torch.long
    ).unsqueeze(0)   # (1, N_eval)

    # --- context phase ---
    h, s = model.context_phase(features_seq, ticker_ids_seq, window_k)
    z = torch.cat([h, s], dim=-1).to(device)   # (1, z_dim)
    z = z.requires_grad_(True)

    # --- decode original ---
    feat_orig = model.decode_features(z.detach(), eval_ids.to(device))[0].cpu().numpy()

    # --- perturbation: gradient of target ticker log_attn w.r.t. z ---
    # We want to shift z in the direction that increases log_attn for target_ticker
    with torch.enable_grad():
        z_pert = z.clone().detach().requires_grad_(True)
        target_ids = torch.tensor([[target_id]], dtype=torch.long, device=device)
        feat_target = model.decode_features(z_pert, target_ids)   # (1, 1, 5)
        log_attn_target = feat_target[0, 0, 0]                    # scalar
        log_attn_target.backward()
        grad = z_pert.grad.clone()

    # Normalise gradient and apply delta
    grad_norm = grad.norm() + 1e-8
    z_perturbed = (z.detach() + delta_log_attn * grad / grad_norm).to(device)

    # --- decode perturbed ---
    with torch.no_grad():
        feat_pert = model.decode_features(z_perturbed, eval_ids.to(device))[0].cpu().numpy()

    # --- build results DataFrame ---
    feat_names = ["log_attn", "bullish_rate", "bearish_rate", "unlabeled_rate", "attn_growth"]
    rows = []
    for i, ticker in enumerate(eval_tickers):
        for j, feat in enumerate(feat_names):
            rows.append({
                "ticker":    ticker,
                "feat":      feat,
                "original":  float(feat_orig[i, j]),
                "perturbed": float(feat_pert[i, j]),
                "delta":     float(feat_pert[i, j] - feat_orig[i, j]),
            })

    return pd.DataFrame(rows)


def plot_counterfactual(
    df: pd.DataFrame,
    target_ticker: str,
    feat: str = "log_attn",
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Bar chart of Δlog_attn for each tracked ticker."""
    sub = df[df["feat"] == feat].copy()
    sub = sub.sort_values("delta", ascending=True)

    colors = ["#D62728" if t == target_ticker else
              ("#2CA02C" if v > 0 else "#1F77B4")
              for t, v in zip(sub["ticker"], sub["delta"])]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sub["ticker"], sub["delta"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Δ {feat} after perturbing {target_ticker}")
    ax.set_title(f"Counterfactual: spike {target_ticker} → effect on other tickers")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
