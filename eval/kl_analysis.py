"""
KL divergence spike analysis.

Loads the KL log produced by the Trainer and plots KL_t over time,
aligned with known regime-transition events (COVID, GME, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


KNOWN_EVENTS = [
    {"date": "2020-02-20", "label": "COVID onset",      "color": "#D62728"},
    {"date": "2020-03-23", "label": "COVID market low", "color": "#FF7F0E"},
    {"date": "2021-01-22", "label": "GME squeeze",      "color": "#9467BD"},
    {"date": "2021-01-28", "label": "GME peak",         "color": "#8C564B"},
    {"date": "2020-11-09", "label": "Vaccine announcement", "color": "#2CA02C"},
]


def load_kl_log(log_path: str | Path) -> pd.DataFrame:
    with open(log_path) as f:
        entries = json.load(f)
    rows = []
    for e in entries:
        rows.append({
            "step":    e["step"],
            "epoch":   e["epoch"],
            "kl_mean": e["kl_mean"],
            "kl_max":  e["kl_max"],
        })
    return pd.DataFrame(rows)


def plot_kl_timeline(
    kl_df: pd.DataFrame,
    week_index: list[str] | None = None,   # list of ISO dates aligned to steps
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot KL_t over training steps (or weeks if week_index provided).

    Parameters
    ----------
    week_index : if provided, maps steps to calendar dates for event alignment
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    x = kl_df["step"].values
    ax.plot(x, kl_df["kl_mean"].values, lw=1.5, color="#1f77b4", label="KL mean")
    ax.fill_between(x, 0, kl_df["kl_max"].values, alpha=0.15, color="#1f77b4")

    ax.set_xlabel("Training step")
    ax.set_ylabel("KL divergence")
    ax.set_title("KL_t over training — regime surprise signal")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_kl_vs_time(
    kl_series: np.ndarray,   # (T,) — one value per week
    week_dates: list[str],   # ISO date strings, len == T
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot KL_t aligned to calendar dates with known event annotations.
    Use this for eval-time KL (test1/test2 splits).
    """
    dates = pd.to_datetime(week_dates)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, kl_series, lw=2.0, color="#1f77b4", label="KL divergence")

    for ev in KNOWN_EVENTS:
        ev_date = pd.to_datetime(ev["date"])
        if dates.min() <= ev_date <= dates.max():
            ax.axvline(ev_date, color=ev["color"], lw=1.5, linestyle="--", alpha=0.8)
            ax.text(
                ev_date, ax.get_ylim()[1] * 0.95, ev["label"],
                rotation=90, va="top", ha="right", fontsize=9, color=ev["color"]
            )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Week")
    ax.set_ylabel("KL divergence (prior vs posterior)")
    ax.set_title("Regime surprise signal — KL spikes align with known events")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def compute_spike_stats(
    kl_series: np.ndarray,
    threshold_sigma: float = 2.0,
) -> dict:
    """Find spike indices and their magnitudes (> mean + threshold_sigma * std)."""
    mu, sigma = kl_series.mean(), kl_series.std()
    threshold = mu + threshold_sigma * sigma
    spike_idx = np.where(kl_series > threshold)[0]
    return {
        "mean":       float(mu),
        "std":        float(sigma),
        "threshold":  float(threshold),
        "n_spikes":   len(spike_idx),
        "spike_idx":  spike_idx.tolist(),
        "spike_vals": kl_series[spike_idx].tolist(),
    }


def save_kl_csv(
    kl_series: np.ndarray,
    week_dates: list[str],
    output_path: str | Path,
) -> None:
    df = pd.DataFrame({"week": week_dates, "kl": kl_series})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[kl_analysis] saved → {output_path}")
