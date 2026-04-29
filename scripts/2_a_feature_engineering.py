"""
Step 2a: Feature Engineering

Reads raw StockTwits parquet files, computes the 5 features per (ticker, week),
keeps the top-K tickers by message count each week, builds the vocabulary, and
saves the processed panel to disk.

Usage:
    python scripts/2_a_feature_engineering.py \
        --raw_dir data/raw \
        --out_dir data/processed \
        --top_k 100 \
        --min_weeks 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from data.features import build_feature_panel
from data.vocab import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature engineering for Twit Wave")
    p.add_argument("--raw_dir",  type=str, default="data/raw",       help="Directory with raw parquet files")
    p.add_argument("--out_dir",  type=str, default="data/processed",  help="Output directory")
    p.add_argument("--top_k",   type=int, default=100,               help="Top-K tickers per week by msg count")
    p.add_argument("--min_weeks", type=int, default=10,               help="Min weeks a ticker must appear to enter vocab")
    p.add_argument("--parquet_glob", type=str, default="*.parquet",   help="Glob for raw parquet files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. build feature panel ──────────────────────────────────────────────
    log.info("Building feature panel from %s", raw_dir)
    panel = build_feature_panel(
        parquet_dir=raw_dir,
        parquet_glob=args.parquet_glob,
        top_k=args.top_k,
    )
    log.info("Panel shape: %s  |  weeks: %d  |  unique tickers: %d",
             panel.shape, panel["week"].nunique(), panel["symbol"].nunique())

    # ── 2. temporal splits ───────────────────────────────────────────────────
    panel["week"] = pd.to_datetime(panel["week"])
    train_mask = panel["week"] <  "2019-01-01"
    val_mask   = (panel["week"] >= "2019-01-01") & (panel["week"] < "2020-01-01")
    test1_mask = (panel["week"] >= "2020-01-01") & (panel["week"] < "2020-07-01")
    test2_mask = (panel["week"] >= "2020-10-01") & (panel["week"] < "2021-07-01")

    panel_train = panel[train_mask].copy()
    panel_val   = panel[val_mask].copy()
    panel_test1 = panel[test1_mask].copy()
    panel_test2 = panel[test2_mask].copy()

    log.info("Split sizes — train: %d  val: %d  test1(COVID): %d  test2(GME): %d",
             len(panel_train), len(panel_val), len(panel_test1), len(panel_test2))

    # ── 3. build vocabulary from training tickers ────────────────────────────
    log.info("Building vocabulary (min_weeks=%d)", args.min_weeks)
    # Only include tickers that appear at least min_weeks times in training data
    ticker_counts = panel_train.groupby("symbol")["week"].nunique()
    eligible = ticker_counts[ticker_counts >= args.min_weeks].index.tolist()
    log.info("Eligible tickers in vocab: %d", len(eligible))

    vocab = Vocabulary.build(eligible)
    vocab.save(out_dir / "vocab.json")
    log.info("Vocabulary size (including PAD): %d", len(vocab))

    # ── 4. save processed splits ─────────────────────────────────────────────
    panel.to_parquet(out_dir / "panel_all.parquet", index=False)
    panel_train.to_parquet(out_dir / "panel_train.parquet", index=False)
    panel_val.to_parquet(out_dir / "panel_val.parquet",   index=False)
    panel_test1.to_parquet(out_dir / "panel_test1.parquet", index=False)
    panel_test2.to_parquet(out_dir / "panel_test2.parquet", index=False)

    # ── 5. save summary stats ────────────────────────────────────────────────
    stats = {
        "top_k":         args.top_k,
        "min_weeks":     args.min_weeks,
        "vocab_size":    len(vocab),
        "n_train_rows":  len(panel_train),
        "n_val_rows":    len(panel_val),
        "n_test1_rows":  len(panel_test1),
        "n_test2_rows":  len(panel_test2),
        "train_weeks":   panel_train["week"].nunique(),
        "val_weeks":     panel_val["week"].nunique(),
        "test1_weeks":   panel_test1["week"].nunique(),
        "test2_weeks":   panel_test2["week"].nunique(),
        "feature_cols":  ["log_attention", "bullish_rate", "bearish_rate", "unlabeled_rate", "attn_growth"],
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Feature engineering complete. Outputs written to %s", out_dir)
    log.info("Stats: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
