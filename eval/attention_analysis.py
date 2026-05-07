"""
Cross-ticker self-attention matrix analysis.

Analyses how the Stage-1 attention structure evolves over time.
Diagonal mass  → intrinsic appeal proxy  (ticker attends to itself)
Off-diagonal   → extrinsic coupling      (ticker attends to peers)

Public API (used by 3_c_eval_attention.py)
------------------------------------------
diagonal_vs_offdiagonal(A)  — single (N,N) matrix → (diag_mean, offdiag_mean)
plot_attention_heatmap      — seaborn heatmap for one week
plot_attention_evolution    — coupling-ratio time series with event annotations
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch

from model.twit_wave import TwitWave
from data.vocab import Vocabulary


# ------------------------------------------------------------------
# matrix stats
# ------------------------------------------------------------------

def diagonal_vs_offdiagonal(A: np.ndarray) -> tuple[float, float]:
    """
    Compute diagonal mean and off-diagonal mean for a single (N, N) matrix.

    Returns
    -------
    diag_mean    : mean of diagonal entries   (intrinsic / self-attention)
    offdiag_mean : mean of off-diagonal entries (extrinsic / cross-ticker coupling)
    """
    N = A.shape[0]
    if N == 0:
        return 0.0, 0.0
    diag_sum    = float(np.trace(A))
    offdiag_sum = float(A.sum()) - diag_sum
    diag_mean    = diag_sum / N
    offdiag_mean = offdiag_sum / max(N * N - N, 1)
    return diag_mean, offdiag_mean


# ------------------------------------------------------------------
# plots
# ------------------------------------------------------------------

def plot_attention_heatmap(
    A: np.ndarray,
    ticker_labels: list[str],
    title: str = "",
    output_path: str | Path | None = None,
    show: bool = True,
    top_n: int = 20,
) -> plt.Figure:
    """Seaborn heatmap for one week's cross-ticker attention matrix."""
    n      = min(len(ticker_labels), top_n, A.shape[0])
    A_sub  = A[:n, :n]
    labels = ticker_labels[:n]

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        A_sub,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
        vmin=0,
        linewidths=0.3,
        cbar_kws={"label": "Attention weight"},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Source ticker")
    ax.set_ylabel("Target ticker")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_attention_evolution(
    coupling_df: pd.DataFrame,
    known_events: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot diagonal vs off-diagonal attention mass over time, with optional
    vertical annotations for known market events.

    Parameters
    ----------
    coupling_df   : DataFrame with columns week, diag_mean, offdiag_mean,
                    coupling_ratio  (produced by 3_c_eval_attention.py)
    known_events  : {date_str: label} dict — dates get vertical dashed lines
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    weeks = pd.to_datetime(coupling_df["week"])

    axes[0].plot(weeks, coupling_df["diag_mean"],    lw=2, color="#2ca02c", label="Diagonal (intrinsic)")
    axes[0].plot(weeks, coupling_df["offdiag_mean"], lw=2, color="#d62728", label="Off-diagonal (extrinsic)")
    axes[0].set_ylabel("Mean attention mass")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)

    axes[1].plot(weeks, coupling_df["coupling_ratio"], lw=2, color="#1f77b4")
    axes[1].axhline(1.0, color="gray", lw=1, linestyle="--", label="ratio = 1")
    axes[1].set_ylabel("Intrinsic / Extrinsic ratio")
    axes[1].grid(alpha=0.2)

    # Annotate known events
    if known_events:
        for date_str, label in known_events.items():
            dt = pd.Timestamp(date_str)
            for ax in axes:
                ax.axvline(dt, color="orange", lw=1.2, linestyle="--", alpha=0.8)
            axes[0].text(dt, axes[0].get_ylim()[1] * 0.97, label,
                         rotation=90, va="top", ha="right", fontsize=7, color="orange")

    axes[0].set_title("Cross-ticker attention structure over time")
    axes[1].xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%Y-%m")
    )
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
