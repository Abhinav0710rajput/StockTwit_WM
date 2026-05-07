"""
Step 3b: KL-divergence regime-transition analysis

Plots the KL(q‖p) time series, annotates known market events (COVID crash,
GME squeeze, etc.), computes spike statistics, and saves a CSV of weekly KL values.

The KL timeline is the primary qualitative diagnostic for the RSSM:
  - Spikes indicate the model's latent regime is shifting rapidly
  - If spikes align with known shocks, it validates the world model hypothesis

Usage:
    python scripts/3_b_eval_kl.py \
        --model_dir outputs/rssm_base \
        --data_dir  data/processed_week \
        --out_dir   outputs/eval/kl
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
from eval.kl_analysis import (
    load_kl_log,
    plot_kl_timeline,
    compute_spike_stats,
    save_kl_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  type=str, default="outputs/rssm_base")
    p.add_argument("--data_dir",   type=str, default="data/processed_week")
    p.add_argument("--out_dir",    type=str, default=None,
                   help="Output directory. Defaults to <model_dir>/results/kl")
    p.add_argument("--splits",     nargs="+", default=["val", "test1", "test2"])
    p.add_argument("--context_len", type=int, default=52)
    p.add_argument("--spike_z",     type=float, default=2.0,
                   help="Z-score threshold for spike detection")
    p.add_argument("--use_train_kl", action="store_true",
                   help="Also plot KL from the training log (kl_log.json)")
    return p.parse_args()


@torch.no_grad()
def compute_weekly_kl(
    model: TwitWave,
    panel: pd.DataFrame,
    vocab: Vocabulary,
    device: torch.device,
    context_len: int,
    norm_stats: dict,
) -> pd.DataFrame:
    """
    For each week in panel (after a burn-in), compute the KL(q‖p) at that step.
    Returns DataFrame with columns: week, kl_mean, kl_sum, kl_max_dim.
    """
    from eval.predict import Predictor
    predictor = Predictor(model=model, device=device, norm_stats=norm_stats)

    weeks = sorted(panel["week"].unique())
    rows  = []

    # We process the panel as a single sequence using the predictor's context builder
    for i, week in enumerate(weeks[context_len:], start=context_len):
        ctx_weeks = weeks[max(0, i - context_len):i]
        ctx_panel = panel[panel["week"].isin(ctx_weeks)].sort_values("week")

        try:
            feat_seq, ids_seq = predictor.build_context(ctx_panel, vocab, max_len=context_len)
        except Exception:
            continue

        # Run model in posterior mode to get KL at the last step
        kl_t = predictor.compute_kl_sequence(feat_seq, ids_seq)
        if kl_t is None or len(kl_t) == 0:
            continue

        kl_last = float(kl_t[-1])
        rows.append({"week": week, "kl": kl_last})

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.model_dir) / "results" / "kl"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    vocab = Vocabulary.load(data_dir / "vocab.json")
    panel_train = pd.read_parquet(data_dir / "panel_train.parquet")

    with open(Path(args.model_dir) / "norm_stats.json") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    log.info("Loading TwitWave from %s", args.model_dir)
    model = load_rssm(Path(args.model_dir), len(vocab), device)

    # ── Option 1: use KL log written during training ──────────────────────────
    if args.use_train_kl:
        kl_log_path = Path(args.model_dir) / "logs" / "kl_log.json"
        if kl_log_path.exists():
            kl_df = load_kl_log(kl_log_path)
            log.info("Loaded training KL log: %d epochs", len(kl_df))
        else:
            log.warning("kl_log.json not found at %s", kl_log_path)

    # ── Option 2: compute KL for each split ───────────────────────────────────
    all_kl: list[pd.DataFrame] = []

    for split in args.splits:
        panel_path = data_dir / f"panel_{split}.parquet"
        if not panel_path.exists():
            log.warning("%s not found, skipping", panel_path)
            continue
        panel = pd.read_parquet(panel_path)
        # Prepend enough training data for context warm-up
        ctx_train = panel_train.tail(args.context_len * 200)   # row budget
        full_panel = pd.concat([ctx_train, panel]).drop_duplicates(subset=["week", "symbol"])

        log.info("Computing weekly KL for %s (%d weeks)", split, panel["week"].nunique())
        kl_df = compute_weekly_kl(
            model=model,
            panel=full_panel,
            vocab=vocab,
            device=device,
            context_len=args.context_len,
            norm_stats=norm_stats,
        )
        kl_df["split"] = split
        all_kl.append(kl_df)

        # Per-split save
        save_kl_csv(kl_df, out_dir / f"kl_{split}.csv")
        spike_stats = compute_spike_stats(kl_df["kl"].values, z_threshold=args.spike_z)
        with open(out_dir / f"spike_stats_{split}.json", "w") as f:
            json.dump(spike_stats, f, indent=2)
        log.info("[%s] Spikes (z>%.1f): %d  |  Max KL=%.4f  |  Mean KL=%.4f",
                 split, args.spike_z,
                 spike_stats["n_spikes"], spike_stats["max_kl"], spike_stats["mean_kl"])

    if not all_kl:
        log.error("No KL data computed — check split panels exist.")
        return

    full_kl = pd.concat(all_kl, ignore_index=True)
    save_kl_csv(full_kl, out_dir / "kl_all_splits.csv")

    # ── Re-compute spike stats using val baseline for cross-split z-scores ────
    # Per-split z-scores are relative to the split's own mean/std, which can
    # miss spikes if the entire split is elevated (e.g., all of test1 is COVID).
    # Use val KL mean/std as the stable-baseline reference instead.
    val_kl_df = next((d for d in all_kl if d["split"].iloc[0] == "val"), None)
    if val_kl_df is not None and len(val_kl_df) > 1:
        val_mean = val_kl_df["kl"].mean()
        val_std  = val_kl_df["kl"].std() + 1e-8
        log.info("Val KL baseline: mean=%.4f  std=%.4f", val_mean, val_std)
        for split_df in all_kl:
            sp = split_df["split"].iloc[0]
            z_scores = (split_df["kl"].values - val_mean) / val_std
            n_spikes = int((z_scores > args.spike_z).sum())
            spike_weeks = split_df["week"].values[z_scores > args.spike_z].tolist()
            baseline_stats = {
                "baseline": "val",
                "val_mean": float(val_mean),
                "val_std":  float(val_std),
                "n_spikes_vs_val": n_spikes,
                "spike_weeks_vs_val": [str(w) for w in spike_weeks],
                "max_z_vs_val": float(z_scores.max()),
            }
            path = out_dir / f"spike_stats_{sp}_vs_val.json"
            with open(path, "w") as f:
                json.dump(baseline_stats, f, indent=2)
            log.info("[%s vs val] Spikes (z>%.1f): %d  max_z=%.2f  weeks=%s",
                     sp, args.spike_z, n_spikes, float(z_scores.max()),
                     spike_weeks[:5])

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("Generating KL timeline plots")
    fig = plot_kl_timeline(
        weeks=full_kl["week"].tolist(),
        kl_values=full_kl["kl"].tolist(),
        output_path=out_dir / "kl_timeline.pdf",
        show=False,
    )
    fig.savefig(out_dir / "kl_timeline.png", dpi=150, bbox_inches="tight")

    log.info("KL analysis complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
