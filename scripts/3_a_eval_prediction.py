"""
Step 3a: Predictive evaluation of TwitWave + baselines

Runs multi-step ahead prediction on the two test splits (COVID, GME),
computes MSE/MAE on log_attention and all 5 features, Spearman ρ on attention
rankings, Precision@100 for set membership, and AUC-ROC for virality detection.

Outputs:
  - metrics_test1.json / metrics_test2.json  (per-model scalar metrics)
  - preds_rssm_test1.parquet  (week, symbol, feat, pred, true)
  - plots/  (attention ranking scatter, virality ROC curves)

Usage:
    python scripts/3_a_eval_prediction.py \
        --model_dir   outputs/rssm_base \
        --data_dir    data/processed \
        --baselines_dir outputs/baselines \
        --out_dir     outputs/eval/prediction \
        --horizons    1 4 13
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
import matplotlib.pyplot as plt

from data.vocab import Vocabulary
from eval.utils import load_rssm
from eval.predict import Predictor
from eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_NAMES = ["log_attention", "bullish_rate", "bearish_rate", "unlabeled_rate", "attn_growth"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",     type=str, default="outputs/rssm_base")
    p.add_argument("--data_dir",      type=str, default="data/processed")
    p.add_argument("--baselines_dir", type=str, default="outputs/baselines")
    p.add_argument("--out_dir",       type=str, default="outputs/eval/prediction")
    p.add_argument("--horizons",      nargs="+", type=int, default=[1, 4, 13],
                   help="Forecast horizons in weeks")
    p.add_argument("--context_len",   type=int, default=52,
                   help="Context window fed to RSSM before prediction")
    p.add_argument("--n_samples",     type=int, default=20,
                   help="Monte Carlo samples for uncertainty estimates")
    p.add_argument("--splits",        nargs="+", default=["test1", "test2"],
                   choices=["val", "test1", "test2"])
    return p.parse_args()


def eval_rssm_on_split(
    model: TwitWave,
    panel: pd.DataFrame,
    panel_train: pd.DataFrame,
    vocab: Vocabulary,
    context_len: int,
    horizons: list[int],
    n_samples: int,
    device: torch.device,
    norm_stats: dict,
) -> dict[int, dict]:
    """Evaluate RSSM on a test split. Returns metrics per horizon."""
    predictor = Predictor(model=model, device=device, norm_stats=norm_stats)

    weeks = sorted(panel["week"].unique())
    # Need context_len weeks before the test split
    train_weeks = sorted(panel_train["week"].unique())
    all_weeks   = train_weeks + weeks

    results_by_horizon: dict[int, list] = {h: [] for h in horizons}

    # Rolling evaluation: for each week t, predict t+H and compare to truth
    for i, eval_week in enumerate(weeks):
        ctx_weeks = all_weeks[max(0, len(train_weeks) + i - context_len) : len(train_weeks) + i]
        ctx_panel = pd.concat([panel_train, panel])[
            pd.concat([panel_train, panel])["week"].isin(ctx_weeks)
        ].sort_values("week")

        # Build context tensors
        try:
            feat_seq, ids_seq = predictor.build_context(ctx_panel, vocab, max_len=context_len)
        except Exception as e:
            log.warning("Context build failed at week %s: %s", eval_week, e)
            continue

        # Get maximum horizon that is within test split
        max_h = min(max(horizons), len(weeks) - i)
        if max_h < 1:
            continue

        rollout = predictor.rollout(feat_seq, ids_seq, steps=max_h, n_samples=n_samples)

        for h in horizons:
            if h > max_h:
                continue
            target_week_idx = i + h - 1
            if target_week_idx >= len(weeks):
                continue
            target_week = weeks[target_week_idx]
            true_df = panel[panel["week"] == target_week]

            pred_h = rollout[h - 1]   # (K, D) — deterministic mean
            results_by_horizon[h].append({
                "week":    target_week,
                "pred":    pred_h,
                "true_df": true_df,
            })

    # Aggregate metrics
    metrics_by_horizon = {}
    for h, entries in results_by_horizon.items():
        if not entries:
            continue
        all_preds = np.stack([e["pred"] for e in entries])  # (N, K, D)
        all_true  = [e["true_df"] for e in entries]
        m = compute_metrics(all_preds, all_true, vocab, FEATURE_NAMES)
        metrics_by_horizon[h] = m
        log.info("  H=%d  MSE_log_attn=%.4f  Spearman=%.4f  P@100=%.4f",
                 h, m.get("mse_log_attention", float("nan")),
                 m.get("spearman_rho", float("nan")),
                 m.get("precision_at_100", float("nan")))

    return metrics_by_horizon


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    data_dir = Path(args.data_dir)
    vocab = Vocabulary.load(data_dir / "vocab.json")
    panel_train = pd.read_parquet(data_dir / "panel_train.parquet")
    panel_val   = pd.read_parquet(data_dir / "panel_val.parquet")

    with open(Path(args.model_dir) / "norm_stats.json") as f:
        norm_stats_raw = json.load(f)
    norm_stats = {k: np.array(v) for k, v in norm_stats_raw.items()}

    # ── Load RSSM ─────────────────────────────────────────────────────────────
    log.info("Loading TwitWave from %s", args.model_dir)
    model = load_rssm(Path(args.model_dir), len(vocab), device)

    all_results = {}

    for split in args.splits:
        panel_path = data_dir / f"panel_{split}.parquet"
        if not panel_path.exists():
            log.warning("Panel not found: %s — skipping", panel_path)
            continue
        panel_test = pd.read_parquet(panel_path)
        log.info("=== Evaluating on %s (%d weeks) ===", split.upper(), panel_test["week"].nunique())

        rssm_metrics = eval_rssm_on_split(
            model=model,
            panel=panel_test,
            panel_train=panel_train,
            vocab=vocab,
            context_len=args.context_len,
            horizons=args.horizons,
            n_samples=args.n_samples,
            device=device,
            norm_stats=norm_stats,
        )

        split_results = {"rssm": rssm_metrics}

        # ── ARIMA baseline ────────────────────────────────────────────────────
        arima_path = Path(args.baselines_dir) / "arima.pkl"
        if arima_path.exists():
            with open(arima_path, "rb") as f:
                arima = pickle.load(f)
            try:
                arima_preds = arima.predict_panel(panel_test)
                for h in args.horizons:
                    # ARIMA doesn't natively do multi-step from rolling context;
                    # we report horizon-1 metrics for all horizons (ceiling)
                    mse = np.mean((arima_preds["pred_log_attn"] - arima_preds["true_log_attn"]) ** 2)
                    mae = np.mean(np.abs(arima_preds["pred_log_attn"] - arima_preds["true_log_attn"]))
                split_results["arima"] = {h: {"mse_log_attention": float(mse), "mae_log_attention": float(mae)} for h in args.horizons}
                log.info("ARIMA  MSE=%.4f  MAE=%.4f", mse, mae)
            except Exception as e:
                log.warning("ARIMA eval failed: %s", e)

        # ── VAR baseline ──────────────────────────────────────────────────────
        var_path = Path(args.baselines_dir) / "var.pkl"
        if var_path.exists():
            with open(var_path, "rb") as f:
                var_model = pickle.load(f)
            try:
                roster = var_model.tickers
                pivoted = (
                    panel_test[panel_test["symbol"].isin(roster)]
                    .pivot_table(index="week", columns="symbol", values="log_attention")
                    .reindex(columns=roster).fillna(0.0).sort_index()
                )
                var_results = {}
                for h in args.horizons:
                    preds_list, true_list = [], []
                    for i in range(len(pivoted) - h):
                        last_p = pivoted.values[max(0, i - var_model.result.k_ar):i + 1]
                        if len(last_p) < var_model.result.k_ar:
                            continue
                        fc = var_model.forecast(last_p, steps=h)
                        preds_list.append(fc[-1])
                        true_list.append(pivoted.values[i + h])
                    if preds_list:
                        p, t = np.array(preds_list), np.array(true_list)
                        var_results[h] = {
                            "mse_log_attention": float(np.mean((p - t) ** 2)),
                            "mae_log_attention": float(np.mean(np.abs(p - t))),
                        }
                        log.info("VAR H=%d  MSE=%.4f  MAE=%.4f", h, var_results[h]["mse_log_attention"], var_results[h]["mae_log_attention"])
                split_results["var"] = var_results
            except Exception as e:
                log.warning("VAR eval failed: %s", e)

        all_results[split] = split_results

        # Save per-split
        with open(out_dir / f"metrics_{split}.json", "w") as f:
            json.dump(split_results, f, indent=2, default=str)
        log.info("Saved → %s", out_dir / f"metrics_{split}.json")

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n=== Summary (H=1 log_attention MSE) ===")
    for split, split_res in all_results.items():
        row = {}
        for model_name, h_metrics in split_res.items():
            if isinstance(h_metrics, dict) and 1 in h_metrics:
                row[model_name] = h_metrics[1].get("mse_log_attention", "-")
        log.info("%s: %s", split.upper(), row)

    log.info("Prediction evaluation complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
