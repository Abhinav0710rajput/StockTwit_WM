"""
BCE residual cross-ticker correlation.

Tests whether z_t acts as a sufficient statistic for the joint ticker distribution.
If the Bernoulli decoder is well-calibrated and z_t captures all shared variation,
residuals r_i = y_i - σ(logit_i) should be approximately uncorrelated across tickers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from model.twit_wave import TwitWave


def compute_residuals(
    presence_logits: np.ndarray,   # (T, vocab_size)
    presence_true: np.ndarray,     # (T, vocab_size)
) -> np.ndarray:
    """r_{i,t} = y_{i,t} - σ(logit_{i,t})"""
    probs = 1.0 / (1.0 + np.exp(-presence_logits))   # sigmoid
    return presence_true - probs                       # (T, vocab_size)


def compute_residual_correlation(
    residuals: np.ndarray,          # (T, vocab_size)
    active_tickers: list[int],      # indices to include (those ever active)
    top_n: int = 50,
) -> np.ndarray:
    """
    Compute Pearson correlation matrix of residuals across active tickers.
    Returns (top_n, top_n) correlation matrix.
    """
    sub = residuals[:, active_tickers[:top_n]]   # (T, top_n)
    # Remove tickers with zero variance
    std = sub.std(axis=0)
    valid = std > 1e-6
    sub = sub[:, valid]
    corr = np.corrcoef(sub.T)                    # (top_n, top_n)
    return corr


def mean_abs_offdiagonal(corr: np.ndarray) -> float:
    """Mean absolute off-diagonal correlation — should be near 0 if z_t is sufficient."""
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(corr[mask]).mean())


def plot_residual_heatmap(
    corr: np.ndarray,
    ticker_labels: list[str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        xticklabels=ticker_labels,
        yticklabels=ticker_labels,
        ax=ax,
    )
    ax.set_title("BCE residual cross-ticker correlation\n(near zero → z_t is sufficient statistic)")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def save_correlation_csv(
    corr: np.ndarray,
    ticker_labels: list[str],
    output_path: str | Path,
) -> None:
    df = pd.DataFrame(corr, index=ticker_labels, columns=ticker_labels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"[residual_correlation] saved → {output_path}")
