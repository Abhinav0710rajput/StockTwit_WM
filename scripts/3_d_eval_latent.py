"""
Step 3d: Latent state clustering and dimensionality reduction

Extracts the latent state z_t = [h_t; s_t] for every week,
reduces to 2D via t-SNE and UMAP, fits k-means and GMM clusters,
and visualises whether the model learns distinct market eras
(pre-2016 calm / 2016-2019 vol / COVID / GME / recovery).

Usage:
    python scripts/3_d_eval_latent.py \
        --model_dir outputs/rssm_base \
        --data_dir  data/processed \
        --out_dir   outputs/eval/latent \
        --n_clusters 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data.vocab import Vocabulary
from eval.utils import load_rssm
from eval.latent_clustering import (
    extract_latent_states,
    run_tsne,
    run_umap,
    fit_clusters,
    silhouette_vs_era,
    plot_latent_2d,
    save_latent_csv,
    ERA_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  type=str, default="outputs/rssm_base")
    p.add_argument("--data_dir",   type=str, default="data/processed")
    p.add_argument("--out_dir",    type=str, default="outputs/eval/latent")
    p.add_argument("--splits",     nargs="+", default=["train", "val", "test1", "test2"])
    p.add_argument("--n_clusters", type=int, default=5)
    p.add_argument("--context_len", type=int, default=52)
    p.add_argument("--tsne_perp",  type=int, default=30)
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def assign_era(week: pd.Timestamp) -> str:
    w = pd.Timestamp(week)
    for (start, end), label in ERA_LABELS.items():
        if pd.Timestamp(start) <= w < pd.Timestamp(end):
            return label
    return "Other"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    vocab = Vocabulary.load(data_dir / "vocab.json")
    panel_train = pd.read_parquet(data_dir / "panel_train.parquet")

    with open(Path(args.model_dir) / "norm_stats.json") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    model = load_rssm(Path(args.model_dir), len(vocab), device)

    from eval.predict import Predictor
    predictor = Predictor(model=model, device=device, norm_stats=norm_stats)

    # ── Collect latent states across all splits ───────────────────────────────
    all_z: list[np.ndarray] = []
    all_weeks: list[str]    = []
    all_eras: list[str]     = []
    all_splits: list[str]   = []

    for split in args.splits:
        if split == "train":
            panel = panel_train
        else:
            panel_path = data_dir / f"panel_{split}.parquet"
            if not panel_path.exists():
                log.warning("%s not found, skipping", panel_path)
                continue
            panel = pd.read_parquet(panel_path)

        weeks = sorted(panel["week"].unique())
        log.info("Extracting latent states for %s (%d weeks)...", split, len(weeks))

        for i, week in enumerate(weeks):
            ctx_train_weeks = sorted(panel_train["week"].unique())[-args.context_len:]
            if split == "train":
                ctx_weeks_prior = weeks[max(0, i - args.context_len):i]
                ctx_panel = panel[panel["week"].isin(ctx_weeks_prior)].sort_values("week")
            else:
                ctx_panel = pd.concat([panel_train, panel])[
                    pd.concat([panel_train, panel])["week"].isin(
                        ctx_train_weeks + weeks[:i]
                    )
                ].sort_values("week").tail(args.context_len * 200)

            try:
                feat_seq, ids_seq = predictor.build_context(ctx_panel, vocab, max_len=args.context_len)
                z = predictor.get_last_latent(feat_seq, ids_seq)  # (z_dim,)
                if z is not None:
                    all_z.append(z)
                    all_weeks.append(str(week))
                    all_eras.append(assign_era(week))
                    all_splits.append(split)
            except Exception as e:
                log.debug("Latent extraction failed at %s: %s", week, e)

    if not all_z:
        log.error("No latent states extracted.")
        return

    Z = np.stack(all_z)   # (N, z_dim)
    log.info("Extracted %d latent states of dim %d", len(Z), Z.shape[1])

    # ── Clustering ────────────────────────────────────────────────────────────
    log.info("Fitting k-means (k=%d) and GMM...", args.n_clusters)
    cluster_labels, gmm_labels, silhouette = fit_clusters(Z, n_clusters=args.n_clusters)
    log.info("Silhouette score: %.4f", silhouette)

    # ── Dimensionality reduction ──────────────────────────────────────────────
    log.info("Running t-SNE...")
    Z_tsne = run_tsne(Z, perplexity=args.tsne_perp, random_state=args.seed)

    log.info("Running UMAP...")
    try:
        Z_umap = run_umap(Z, n_neighbors=args.umap_neighbors, random_state=args.seed)
    except ImportError:
        log.warning("umap-learn not installed — skipping UMAP")
        Z_umap = None

    # ── Save latent CSV ───────────────────────────────────────────────────────
    latent_df = pd.DataFrame({
        "week":          all_weeks,
        "era":           all_eras,
        "split":         all_splits,
        "cluster_kmeans": cluster_labels,
        "cluster_gmm":   gmm_labels,
        "tsne_0":        Z_tsne[:, 0],
        "tsne_1":        Z_tsne[:, 1],
    })
    if Z_umap is not None:
        latent_df["umap_0"] = Z_umap[:, 0]
        latent_df["umap_1"] = Z_umap[:, 1]

    save_latent_csv(latent_df, out_dir / "latent_states.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plot_latent_2d(
        Z_2d=Z_tsne,
        labels=all_eras,
        method="t-SNE",
        title="Latent space by market era (t-SNE)",
        output_path=out_dir / "tsne_by_era.png",
        show=False,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    fig = plot_latent_2d(
        Z_2d=Z_tsne,
        labels=[str(l) for l in cluster_labels],
        method="t-SNE",
        title=f"Latent space k-means clusters (k={args.n_clusters})",
        output_path=out_dir / "tsne_by_cluster.png",
        show=False,
    )
    plt.close(fig)

    if Z_umap is not None:
        fig = plot_latent_2d(
            Z_2d=Z_umap,
            labels=all_eras,
            method="UMAP",
            title="Latent space by market era (UMAP)",
            output_path=out_dir / "umap_by_era.png",
            show=False,
        )
        plt.close(fig)

    # ── Era silhouette ────────────────────────────────────────────────────────
    era_sil = silhouette_vs_era(Z, all_eras)
    with open(out_dir / "era_silhouette.json", "w") as f:
        json.dump({
            "silhouette_score": silhouette,
            "n_clusters": args.n_clusters,
            "era_distribution": latent_df["era"].value_counts().to_dict(),
            **era_sil,
        }, f, indent=2)

    log.info("Latent clustering complete. Outputs in %s", out_dir)
    log.info("Silhouette: %.4f", silhouette)


if __name__ == "__main__":
    main()
