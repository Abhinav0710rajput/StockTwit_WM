"""
Latent state clustering analysis.

Extracts z_t for all time steps, runs UMAP + t-SNE,
fits k-means / GMM, computes silhouette scores vs known eras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import torch

from model.twit_wave import TwitWave

ERA_LABELS = {
    ("2008-01-01", "2012-12-31"): ("Early growth",    0),
    ("2013-01-01", "2016-12-31"): ("Maturity",         1),
    ("2017-01-01", "2019-12-31"): ("Pre-COVID",        2),
    ("2020-01-01", "2020-12-31"): ("COVID",            3),
    ("2021-01-01", "2021-12-31"): ("Meme era",         4),
    ("2022-01-01", "2022-12-31"): ("Post-meme",        5),
}
ERA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def assign_era(date: pd.Timestamp) -> int:
    for (lo, hi), (_, idx) in ERA_LABELS.items():
        if pd.Timestamp(lo) <= date <= pd.Timestamp(hi):
            return idx
    return -1


def extract_latent_states(
    model: TwitWave,
    features_seq: torch.Tensor,    # (T, N, 5)
    ticker_ids_seq: torch.Tensor,  # (T, N)
    device: torch.device,
    window_k: int | None = None,
) -> np.ndarray:
    """
    Run the full RSSM forward pass and collect z_t = [h_t, s_t] at each step.
    Returns (T, z_dim).
    """
    if window_k is None:
        window_k = model.cfg.window_k

    model.eval()
    T = features_seq.shape[0]
    features_seq   = features_seq.to(device)
    ticker_ids_seq = ticker_ids_seq.to(device)

    h, s = model.rssm.init_state(1, device)
    a_cache: list[torch.Tensor] = []
    z_list = []

    with torch.no_grad():
        for t in range(T):
            feat = features_seq[t].unsqueeze(0)
            ids  = ticker_ids_seq[t].unsqueeze(0)
            a_t, _ = model._encode_step(feat, ids)
            a_cache.append(a_t)
            window = model._build_window(a_cache, window_k, device)
            e_t = model.temporal_enc(window)
            h = model.rssm.gru_step(h, s)
            s, _, _ = model.rssm.posterior(h, e_t)
            z = torch.cat([h, s], dim=-1)   # (1, z_dim)
            z_list.append(z[0].cpu().numpy())

    return np.stack(z_list)   # (T, z_dim)


def run_tsne(z: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1000)
    return tsne.fit_transform(z)


def run_umap(z: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        return reducer.fit_transform(z)
    except ImportError:
        print("[latent_clustering] umap-learn not installed, skipping UMAP")
        return np.zeros((len(z), 2))


def fit_clusters(z: np.ndarray, n_clusters: int = 5) -> dict:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(z)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(z)

    return {
        "kmeans_labels": km_labels,
        "gmm_labels":    gmm_labels,
        "kmeans_inertia": float(km.inertia_),
    }


def silhouette_vs_era(z: np.ndarray, era_labels: np.ndarray) -> float:
    """Silhouette score of z using known era labels."""
    valid = era_labels >= 0
    if valid.sum() < 10:
        return float("nan")
    return float(silhouette_score(z[valid], era_labels[valid]))


def plot_latent_2d(
    z_2d: np.ndarray,
    era_labels: np.ndarray,
    method: str = "UMAP",
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    era_names = {idx: name for (_, _), (name, idx) in ERA_LABELS.items()}

    for era_idx in sorted(set(era_labels)):
        if era_idx < 0:
            continue
        mask = era_labels == era_idx
        name = era_names.get(era_idx, f"Era {era_idx}")
        color = ERA_COLORS[era_idx % len(ERA_COLORS)]
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, s=12, alpha=0.6, label=name)

    ax.set_title(f"Latent states z_t — {method}")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.legend(markerscale=2, fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def save_latent_csv(
    z: np.ndarray,
    z_tsne: np.ndarray,
    z_umap: np.ndarray,
    era_labels: np.ndarray,
    week_dates: list[str],
    output_path: str | Path,
) -> None:
    df = pd.DataFrame({
        "week": week_dates,
        "era":  era_labels,
        "tsne_x": z_tsne[:, 0],
        "tsne_y": z_tsne[:, 1],
        "umap_x": z_umap[:, 0],
        "umap_y": z_umap[:, 1],
    })
    # also save first few z dims
    for i in range(min(10, z.shape[1])):
        df[f"z_{i}"] = z[:, i]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[latent_clustering] saved → {output_path}")
