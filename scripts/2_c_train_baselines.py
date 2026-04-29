"""
Step 2c: Train all baselines (ARIMA, VAR, LSTM)

Baselines use "fixed-roster" data with zero-imputation (constant top-K tickers
chosen by total training-set message count). This lets ARIMA/VAR/LSTM operate
on a static (T, K) or (T, K, D) tensor without handling dynamic membership.

Usage:
    python scripts/2_c_train_baselines.py \
        --data_dir data/processed \
        --out_dir  outputs/baselines \
        --top_k    100

    # train only specific models:
    python scripts/2_c_train_baselines.py --models arima var
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.dataset import TwitWaveDataset, collate_fixed
from data.vocab import Vocabulary
from baselines.arima import PerTickerARIMA
from baselines.var   import ReducedRankVAR
from baselines.lstm  import SharedLSTM, train_lstm_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = ["log_attention", "bullish_rate", "bearish_rate", "unlabeled_rate", "attn_growth"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--out_dir",  type=str, default="outputs/baselines")
    p.add_argument("--top_k",   type=int, default=100)
    p.add_argument("--models",  nargs="+", default=["arima", "var", "lstm"],
                   choices=["arima", "var", "lstm"])
    # ARIMA
    p.add_argument("--arima_order", nargs=3, type=int, default=[2, 0, 1],
                   help="ARIMA(p,d,q) order")
    # VAR
    p.add_argument("--var_maxlags", type=int, default=4)
    p.add_argument("--var_rank",    type=int, default=10)
    # LSTM
    p.add_argument("--lstm_hidden", type=int, default=512)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--lstm_epochs", type=int, default=30)
    p.add_argument("--lstm_lr",     type=float, default=3e-4)
    p.add_argument("--lstm_batch",  type=int, default=32)
    p.add_argument("--seq_len",     type=int, default=52)
    return p.parse_args()


def get_fixed_roster(panel_train: pd.DataFrame, top_k: int) -> list[str]:
    """Select top_k tickers by total message count in training data."""
    # Reverse log_attention: attn = exp(log_attn) - 1  (approx total msgs)
    # We just rank by sum of log_attention as a proxy
    total = (
        panel_train.groupby("symbol")["log_attention"]
        .sum()
        .sort_values(ascending=False)
    )
    return total.head(top_k).index.tolist()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary.load(data_dir / "vocab.json")
    panel_train = pd.read_parquet(data_dir / "panel_train.parquet")
    panel_val   = pd.read_parquet(data_dir / "panel_val.parquet")

    log.info("Building fixed roster (top-%d by training msg count)", args.top_k)
    roster = get_fixed_roster(panel_train, args.top_k)
    with open(out_dir / "fixed_roster.json", "w") as f:
        json.dump(roster, f, indent=2)
    log.info("Fixed roster tickers: %d", len(roster))

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    if "arima" in args.models:
        log.info("=== Training ARIMA(%s) ===", args.arima_order)
        arima = PerTickerARIMA(order=tuple(args.arima_order))
        arima.fit(panel_train, roster)
        with open(out_dir / "arima.pkl", "wb") as f:
            pickle.dump(arima, f)
        log.info("ARIMA saved → %s", out_dir / "arima.pkl")

        # Quick val forecast
        val_pred = arima.predict_panel(panel_val)
        val_pred.to_csv(out_dir / "arima_val_preds.csv", index=False)

        mse = np.mean((val_pred["pred_log_attn"] - val_pred["true_log_attn"]) ** 2)
        mae = np.mean(np.abs(val_pred["pred_log_attn"] - val_pred["true_log_attn"]))
        log.info("ARIMA val  MSE=%.4f  MAE=%.4f", mse, mae)
        with open(out_dir / "arima_val_metrics.json", "w") as f:
            json.dump({"mse": mse, "mae": mae}, f, indent=2)

    # ── VAR ───────────────────────────────────────────────────────────────────
    if "var" in args.models:
        log.info("=== Training VAR(maxlags=%d, rank=%d) ===", args.var_maxlags, args.var_rank)
        var_model = ReducedRankVAR(maxlags=args.var_maxlags, rank=args.var_rank)
        var_model.fit(panel_train, roster)
        with open(out_dir / "var.pkl", "wb") as f:
            pickle.dump(var_model, f)
        log.info("VAR saved → %s", out_dir / "var.pkl")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if "lstm" in args.models:
        log.info("=== Training Shared LSTM ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("LSTM device: %s", device)

        ds_train = TwitWaveDataset(
            panel=panel_train,
            vocab=vocab,
            seq_len=args.seq_len,
            window_k=1,
            mode="fixed",
            split="train",
            normalise=True,
            fixed_roster=roster,
        )
        ds_val = TwitWaveDataset(
            panel=panel_val,
            vocab=vocab,
            seq_len=args.seq_len,
            window_k=1,
            mode="fixed",
            split="val",
            normalise=True,
            norm_stats=ds_train.norm_stats,
            fixed_roster=roster,
        )

        from torch.utils.data import DataLoader
        train_loader = DataLoader(ds_train, batch_size=args.lstm_batch, shuffle=True,
                                  collate_fn=collate_fixed, drop_last=True)
        val_loader   = DataLoader(ds_val,   batch_size=args.lstm_batch, shuffle=False,
                                  collate_fn=collate_fixed)

        lstm = SharedLSTM(
            n_tickers=len(roster),
            feature_dim=5,
            hidden_dim=args.lstm_hidden,
            n_layers=args.lstm_layers,
        )
        lstm = train_lstm_baseline(
            model=lstm,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.lstm_epochs,
            lr=args.lstm_lr,
            device=device,
            output_path=out_dir / "lstm_best.pt",
        )
        log.info("LSTM saved → %s", out_dir / "lstm_best.pt")

        # Save LSTM architecture info
        with open(out_dir / "lstm_cfg.json", "w") as f:
            json.dump({
                "n_tickers":  len(roster),
                "feature_dim": 5,
                "hidden_dim":  args.lstm_hidden,
                "n_layers":    args.lstm_layers,
            }, f, indent=2)

    log.info("Baselines training complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
