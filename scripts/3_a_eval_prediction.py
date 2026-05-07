"""
Step 3a: Predictive evaluation of TwitWave + baselines

Runs rolling multi-step-ahead prediction on test splits (COVID, GME),
computes MSE/MAE on log_attention and all 5 features, Spearman ρ on
attention rankings, Precision@100 for set membership, and AUC-ROC for
virality detection.

Outputs
-------
metrics_{split}.json                   — per-model, per-horizon scalar metrics
plots/spearman_scatter_{split}_H{h}.png — pred vs true log_attn rank scatter
plots/roc_{split}_H{h}.png             — virality AUC-ROC curve
plots/mse_by_horizon_{split}.png        — MSE comparison across models & horizons
plots/metrics_summary_{split}.png       — full metric comparison table as figure

Usage
-----
    python scripts/3_a_eval_prediction.py \\
        --model_dir   outputs/rssm_base \\
        --data_dir    data/processed_week \\
        --baselines_dir outputs/baselines \\
        --horizons    1 4 13 \\
        --splits      test1 test2
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc as sk_auc
from scipy.stats import spearmanr

from data.vocab import Vocabulary
from eval.utils import load_rssm
from eval.predict import Predictor
from eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_NAMES = ["log_attention", "bullish_rate", "bearish_rate", "unlabeled_rate", "attn_growth"]


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",     type=str, default="outputs/rssm_base")
    p.add_argument("--data_dir",      type=str, default="data/processed_week")
    p.add_argument("--baselines_dir", type=str, default="outputs/baselines")
    p.add_argument("--out_dir",       type=str, default=None,
                   help="Output directory. Defaults to <model_dir>/results/prediction")
    p.add_argument("--horizons",      nargs="+", type=int, default=[1, 4, 13])
    p.add_argument("--context_len",   type=int, default=52)
    p.add_argument("--splits",        nargs="+", default=["test1", "test2"],
                   choices=["val", "test1", "test2"])
    return p.parse_args()


# ── evaluation ────────────────────────────────────────────────────────────────

def eval_rssm_on_split(
    predictor: Predictor,
    panel: pd.DataFrame,
    panel_train: pd.DataFrame,
    vocab: Vocabulary,
    context_len: int,
    horizons: list[int],
) -> dict[int, dict]:
    """
    Rolling evaluation on one test split.

    For each week t in the split:
      1. Build context from train + preceding test weeks (posterior warm-up)
      2. Roll forward max(horizons) steps using the prior
      3. Compare predictions at each horizon to ground truth

    Returns metrics keyed by horizon.
    """
    weeks      = sorted(panel["week"].unique())
    train_weeks = sorted(panel_train["week"].unique())
    all_weeks   = train_weeks + weeks

    # per-horizon accumulators
    pred_feat_h:    dict[int, list] = {h: [] for h in horizons}
    true_feat_h:    dict[int, list] = {h: [] for h in horizons}
    pred_probs_h:   dict[int, list] = {h: [] for h in horizons}
    true_pres_h:    dict[int, list] = {h: [] for h in horizons}
    pred_ids_h:     dict[int, list] = {h: [] for h in horizons}
    true_ids_h:     dict[int, list] = {h: [] for h in horizons}

    for i, eval_week in enumerate(weeks):
        ctx_start = max(0, len(train_weeks) + i - context_len)
        ctx_end   = len(train_weeks) + i
        ctx_weeks = all_weeks[ctx_start:ctx_end]

        combined = pd.concat([panel_train, panel])
        ctx_panel = combined[combined["week"].isin(ctx_weeks)].sort_values("week")
        if ctx_panel.empty:
            continue

        try:
            feat_seq, ids_seq = predictor.build_context(ctx_panel, vocab, max_len=context_len)
        except Exception as e:
            log.warning("Context build failed at %s: %s", eval_week, e)
            continue

        h, s = predictor.context_phase(feat_seq, ids_seq)
        max_h = min(max(horizons), len(weeks) - i)
        if max_h < 1:
            continue

        rollout = predictor.rollout(h, s, steps=max_h)

        for h_val in horizons:
            if h_val > max_h:
                continue
            tgt_idx = i + h_val - 1
            if tgt_idx >= len(weeks):
                continue
            tgt_week = weeks[tgt_idx]
            true_df  = panel[panel["week"] == tgt_week]
            if true_df.empty:
                continue

            step = rollout[h_val - 1]

            # predicted features (K, 5)
            pred_feat_h[h_val].append(step["features"])
            pred_probs_h[h_val].append(step["presence_probs"])
            pred_ids_h[h_val].append(step["ticker_ids"])

            # true features — align on predicted ticker set
            top_k    = step["ticker_ids"].shape[0]
            feat_true = np.zeros((top_k, 5), dtype=np.float32)
            for j, idx in enumerate(step["ticker_ids"]):
                sym = vocab.decode(int(idx))
                row = true_df[true_df["symbol"] == sym]
                if not row.empty:
                    feat_true[j] = row[FEATURE_NAMES].values[0]
            true_feat_h[h_val].append(feat_true)

            # true presence vector
            vocab_size = predictor.model.cfg.vocab_size
            pres = np.zeros(vocab_size, dtype=np.float32)
            for sym in true_df["symbol"].values:
                idx = vocab.encode(sym)
                if idx > 0:
                    pres[idx] = 1.0
            true_pres_h[h_val].append(pres)

            # true active ticker ids
            true_ids = np.array([vocab.encode(s) for s in true_df["symbol"].values
                                 if vocab.encode(s) > 0], dtype=np.int64)
            true_ids_h[h_val].append(true_ids[:top_k] if len(true_ids) >= top_k
                                      else np.pad(true_ids, (0, top_k - len(true_ids))))

    # compute metrics per horizon
    metrics_by_horizon: dict[int, dict] = {}
    for h_val in horizons:
        if not pred_feat_h[h_val]:
            continue
        T   = len(pred_feat_h[h_val])
        K   = pred_feat_h[h_val][0].shape[0]
        m = compute_metrics(
            pred_features  = np.stack(pred_feat_h[h_val]),    # (T, K, 5)
            true_features  = np.stack(true_feat_h[h_val]),    # (T, K, 5)
            pred_presence  = np.stack(pred_probs_h[h_val]),   # (T, vocab_size)
            true_presence  = np.stack(true_pres_h[h_val]),    # (T, vocab_size)
            pred_ticker_ids = np.stack(pred_ids_h[h_val]),    # (T, K)
            true_ticker_ids = np.stack(true_ids_h[h_val]),    # (T, K)
        )
        metrics_by_horizon[h_val] = m
        log.info("  H=%d  MSE_log_attn=%.4f  Spearman=%.4f  P@100=%.4f  AUC=%.4f",
                 h_val, m.get("mse_log_attn", float("nan")),
                 m.get("spearman_rho", float("nan")),
                 m.get("precision_at_100", float("nan")),
                 m.get("auc_roc_virality", float("nan")))

    return {
        "metrics":      metrics_by_horizon,
        "pred_feat":    pred_feat_h,
        "true_feat":    true_feat_h,
        "pred_probs":   pred_probs_h,
        "true_pres":    true_pres_h,
        "pred_ids":     pred_ids_h,
        "true_ids":     true_ids_h,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_spearman_scatter(
    pred_feat: list[np.ndarray],    # list of (K,5) arrays
    true_feat: list[np.ndarray],
    horizon: int,
    split: str,
    out_dir: Path,
) -> None:
    """Scatter of predicted vs true log_attention (index 0), one point per ticker-week."""
    pred_la = np.concatenate([f[:, 0] for f in pred_feat])
    true_la = np.concatenate([f[:, 0] for f in true_feat])

    rho, _ = spearmanr(pred_la, true_la)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_la, pred_la, s=4, alpha=0.3, color="#1f77b4", rasterized=True)

    # regression line
    m, b = np.polyfit(true_la, pred_la, 1)
    xl = np.array([true_la.min(), true_la.max()])
    ax.plot(xl, m * xl + b, color="red", lw=1.5, label=f"fit  ρ={rho:.3f}")

    # identity line
    ax.plot(xl, xl, color="gray", lw=1, linestyle="--", label="y = x")

    ax.set_xlabel("True log_attention")
    ax.set_ylabel("Predicted log_attention")
    ax.set_title(f"Spearman rank scatter — {split.upper()}  H={horizon}  (ρ={rho:.3f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = out_dir / f"spearman_scatter_{split}_H{horizon}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_roc_curve(
    pred_probs: list[np.ndarray],   # list of (vocab_size,) arrays
    true_pres:  list[np.ndarray],   # list of (vocab_size,) binary arrays
    horizon: int,
    split: str,
    virality_k: int,
    virality_horizon: int,
    out_dir: Path,
) -> None:
    """
    Virality AUC-ROC: does a ticker enter the top-K within virality_horizon steps?
    One ROC curve per (split, horizon).
    """
    T  = len(pred_probs)
    all_scores, all_labels = [], []

    for t in range(T - virality_horizon):
        future_mask = np.zeros_like(true_pres[t])
        for offset in range(1, virality_horizon + 1):
            if t + offset >= T:
                break
            pres = true_pres[t + offset]
            topk_idx = np.argsort(pres)[-virality_k:]
            future_mask[topk_idx] = 1.0
        all_scores.append(pred_probs[t])
        all_labels.append(future_mask)

    if not all_scores:
        return

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    if labels.sum() == 0 or labels.sum() == len(labels):
        log.warning("ROC: degenerate labels for %s H=%d, skipping plot", split, horizon)
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, color="#d62728", label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Virality AUC-ROC — {split.upper()}  H={horizon}")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = out_dir / f"roc_{split}_H{horizon}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_mse_by_horizon(
    all_model_metrics: dict[str, dict[int, dict]],
    horizons: list[int],
    split: str,
    out_dir: Path,
) -> None:
    """
    Grouped bar chart: MSE on log_attention per model per horizon.
    models: rssm, arima, var (only if data available)
    """
    model_names = list(all_model_metrics.keys())
    x = np.arange(len(horizons))
    width = 0.8 / max(len(model_names), 1)

    colors = {"rssm": "#1f77b4", "arima": "#ff7f0e", "var": "#2ca02c", "lstm": "#9467bd"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model_names):
        mse_vals = []
        for h in horizons:
            m = all_model_metrics[name].get(h, {})
            mse_vals.append(m.get("mse_log_attn", m.get("mse_log_attention", float("nan"))))
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, mse_vals, width, label=name.upper(),
               color=colors.get(name, f"C{i}"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in horizons])
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MSE  (log_attention)")
    ax.set_title(f"MSE by horizon — {split.upper()}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"mse_by_horizon_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_metrics_summary(
    all_model_metrics: dict[str, dict[int, dict]],
    horizons: list[int],
    split: str,
    out_dir: Path,
) -> None:
    """
    Table figure showing all metrics (MSE, MAE, Spearman ρ, P@100, AUC)
    for each model × horizon.
    """
    METRIC_KEYS = ["mse_log_attn", "mae_log_attn", "spearman_rho", "precision_at_100", "auc_roc_virality"]
    METRIC_LABELS = ["MSE log_attn", "MAE log_attn", "Spearman ρ", "Precision@100", "AUC-ROC"]

    rows = []
    for model_name, h_metrics in all_model_metrics.items():
        for h in horizons:
            m = h_metrics.get(h, {})
            row = {"Model": model_name.upper(), "H": h}
            for key, label in zip(METRIC_KEYS, METRIC_LABELS):
                val = m.get(key, m.get(key.replace("log_attn", "log_attention"), float("nan")))
                row[label] = f"{val:.4f}" if not np.isnan(float(val)) else "—"
            rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    n_rows = len(df)

    fig, ax = plt.subplots(figsize=(13, max(3, 0.4 * n_rows + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))

    # header shading
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#2c7bb6")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # alternating row shading
    for i in range(1, n_rows + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            tbl[i, j].set_facecolor(color)

    ax.set_title(f"Prediction metrics — {split.upper()}", fontsize=12, pad=10)
    plt.tight_layout()
    path = out_dir / f"metrics_summary_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ── baselines ─────────────────────────────────────────────────────────────────

def eval_arima_on_split(
    baselines_dir: Path,
    panel_test: pd.DataFrame,
    horizons: list[int],
) -> dict[int, dict] | None:
    arima_path = baselines_dir / "arima.pkl"
    if not arima_path.exists():
        return None
    try:
        with open(arima_path, "rb") as f:
            arima = pickle.load(f)
        preds = arima.predict_panel(panel_test)
        mse = float(np.mean((preds["pred_log_attn"] - preds["true_log_attn"]) ** 2))
        mae = float(np.mean(np.abs(preds["pred_log_attn"] - preds["true_log_attn"])))
        return {h: {"mse_log_attention": mse, "mae_log_attention": mae} for h in horizons}
    except Exception as e:
        log.warning("ARIMA eval failed: %s", e)
        return None


def eval_var_on_split(
    baselines_dir: Path,
    panel_test: pd.DataFrame,
    horizons: list[int],
) -> dict[int, dict] | None:
    var_path = baselines_dir / "var.pkl"
    if not var_path.exists():
        return None
    try:
        with open(var_path, "rb") as f:
            var_model = pickle.load(f)
        roster  = var_model.tickers
        pivoted = (
            panel_test[panel_test["symbol"].isin(roster)]
            .pivot_table(index="week", columns="symbol", values="log_attention")
            .reindex(columns=roster).fillna(0.0).sort_index()
        )
        var_results = {}
        for h in horizons:
            preds_list, true_list = [], []
            for i in range(len(pivoted) - h):
                obs = pivoted.values[max(0, i - var_model.result.k_ar):i + 1]
                if len(obs) < var_model.result.k_ar:
                    continue
                fc = var_model.forecast(obs, steps=h)
                preds_list.append(fc[-1])
                true_list.append(pivoted.values[i + h])
            if preds_list:
                p, t = np.array(preds_list), np.array(true_list)
                var_results[h] = {
                    "mse_log_attention": float(np.mean((p - t) ** 2)),
                    "mae_log_attention": float(np.mean(np.abs(p - t))),
                }
        return var_results
    except Exception as e:
        log.warning("VAR eval failed: %s", e)
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.model_dir) / "results" / "prediction"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    data_dir    = Path(args.data_dir)
    vocab       = Vocabulary.load(data_dir / "vocab.json")
    panel_train = pd.read_parquet(data_dir / "panel_train.parquet")

    with open(Path(args.model_dir) / "norm_stats.json") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    log.info("Loading TwitWave from %s", args.model_dir)
    model = load_rssm(Path(args.model_dir), len(vocab), device)
    predictor = Predictor(model=model, device=device, norm_stats=norm_stats)

    all_results = {}

    for split in args.splits:
        panel_path = data_dir / f"panel_{split}.parquet"
        if not panel_path.exists():
            log.warning("Panel not found: %s — skipping", panel_path)
            continue
        panel_test = pd.read_parquet(panel_path)
        log.info("=== %s (%d weeks) ===", split.upper(), panel_test["week"].nunique())

        # ── RSSM ──────────────────────────────────────────────────────────────
        rssm_result = eval_rssm_on_split(
            predictor   = predictor,
            panel       = panel_test,
            panel_train = panel_train,
            vocab       = vocab,
            context_len = args.context_len,
            horizons    = args.horizons,
        )
        rssm_metrics = rssm_result["metrics"]

        split_metrics: dict[str, dict[int, dict]] = {"rssm": rssm_metrics}

        # ── baselines ─────────────────────────────────────────────────────────
        arima_m = eval_arima_on_split(Path(args.baselines_dir), panel_test, args.horizons)
        if arima_m:
            split_metrics["arima"] = arima_m
            log.info("ARIMA  H=1  MSE=%.4f", arima_m.get(1, {}).get("mse_log_attention", float("nan")))

        var_m = eval_var_on_split(Path(args.baselines_dir), panel_test, args.horizons)
        if var_m:
            split_metrics["var"] = var_m
            log.info("VAR  H=1  MSE=%.4f", var_m.get(1, {}).get("mse_log_attention", float("nan")))

        all_results[split] = split_metrics

        # ── save JSON metrics ─────────────────────────────────────────────────
        with open(out_dir / f"metrics_{split}.json", "w") as f:
            json.dump(split_metrics, f, indent=2, default=str)
        log.info("Saved → %s", out_dir / f"metrics_{split}.json")

        # ── plots per split ───────────────────────────────────────────────────
        for h in args.horizons:
            if rssm_result["pred_feat"][h]:
                plot_spearman_scatter(
                    pred_feat = rssm_result["pred_feat"][h],
                    true_feat = rssm_result["true_feat"][h],
                    horizon   = h,
                    split     = split,
                    out_dir   = plots_dir,
                )
                plot_roc_curve(
                    pred_probs       = rssm_result["pred_probs"][h],
                    true_pres        = rssm_result["true_pres"][h],
                    horizon          = h,
                    split            = split,
                    virality_k       = 20,
                    virality_horizon = 4,
                    out_dir          = plots_dir,
                )

        plot_mse_by_horizon(split_metrics, args.horizons, split, plots_dir)
        plot_metrics_summary(split_metrics, args.horizons, split, plots_dir)

    # ── summary ───────────────────────────────────────────────────────────────
    log.info("\n=== Summary (H=1 log_attention MSE) ===")
    for split, split_res in all_results.items():
        row = {}
        for model_name, h_metrics in split_res.items():
            m = h_metrics.get(1, {})
            row[model_name] = m.get("mse_log_attn", m.get("mse_log_attention", "—"))
        log.info("%s: %s", split.upper(), row)

    log.info("Prediction evaluation complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
