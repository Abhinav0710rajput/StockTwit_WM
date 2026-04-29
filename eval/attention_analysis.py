"""
Cross-ticker self-attention matrix analysis.

Analyses how the Stage-1 attention structure evolves over time.
Diagonal mass → intrinsic appeal proxy.
Off-diagonal mass → extrinsic cross-ticker coupling proxy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from model.twit_wave import TwitWave
from data.vocab import Vocabulary


def extract_attention_matrices(
    model: TwitWave,
    features_seq: torch.Tensor,    # (T, N, 5)
    ticker_ids_seq: torch.Tensor,  # (T, N)
    device: torch.device,
) -> list[np.ndarray]:
    """
    Run Stage-1 encoder on each time step and collect A_t matrices.

    Returns list of T arrays, each (N_t, N_t).
    """
    model.eval()
    A_list = []

    with torch.no_grad():
        for t in range(features_seq.shape[0]):
            feat = features_seq[t].unsqueeze(0).to(device)   # (1, N, 5)
            ids  = ticker_ids_seq[t].unsqueeze(0).to(device)  # (1, N)
            a_t, A_t = model._encode_step(feat, ids)
            A_list.append(A_t[0].cpu().numpy())               # (N, N)

    return A_list


def diagonal_vs_offdiagonal(A_list: list[np.ndarray]) -> pd.DataFrame:
    """
    Compute per-step diagonal mass and off-diagonal mass.
    Returns DataFrame with columns: step, diag_mass, offdiag_mass, ratio.
    """
    rows = []
    for t, A in enumerate(A_list):
        N = A.shape[0]
        if N == 0:
            continue
        diag  = np.trace(A) / N
        total = A.sum() / (N * N)
        offdiag = (total * N * N - np.trace(A)) / max(N * N - N, 1)
        rows.append({
            "step":      t,
            "diag_mass": float(diag),
            "offdiag_mass": float(offdiag),
            "ratio":     float(diag / max(offdiag, 1e-8)),
        })
    return pd.DataFrame(rows)


def plot_attention_heatmap(
    A: np.ndarray,
    ticker_labels: list[str],
    title: str = "",
    output_path: str | Path | None = None,
    show: bool = True,
    top_n: int = 20,   # only show top_n × top_n for readability
) -> plt.Figure:
    """Plot a single attention heatmap."""
    n = min(len(ticker_labels), top_n)
    A_sub = A[:n, :n]
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
    diag_df: pd.DataFrame,
    week_dates: list[str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot diagonal vs off-diagonal mass over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    x = week_dates if week_dates else diag_df["step"].values

    axes[0].plot(x, diag_df["diag_mass"], lw=2, color="#2ca02c", label="Diagonal (intrinsic)")
    axes[0].plot(x, diag_df["offdiag_mass"], lw=2, color="#d62728", label="Off-diagonal (extrinsic)")
    axes[0].set_ylabel("Mean attention mass")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(x, diag_df["ratio"], lw=2, color="#1f77b4")
    axes[1].set_ylabel("Intrinsic / Extrinsic ratio")
    axes[1].axhline(1.0, color="gray", lw=1, linestyle="--")
    axes[1].grid(alpha=0.2)

    axes[0].set_title("Cross-ticker attention structure over time")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
