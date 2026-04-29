"""
Evaluation metrics.

compute_metrics — MSE, MAE, Spearman ρ, Precision@100, AUC-ROC (virality)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compute_metrics(
    pred_features: np.ndarray,      # (T, N, 5)  predicted features
    true_features: np.ndarray,      # (T, N, 5)
    pred_presence: np.ndarray,      # (T, vocab_size) probabilities
    true_presence: np.ndarray,      # (T, vocab_size) binary
    pred_ticker_ids: np.ndarray,    # (T, N) predicted top-100 indices
    true_ticker_ids: np.ndarray,    # (T, N) actual top-100 indices
    log_attn_idx: int = 0,          # index of log_attention in feature dim
    top_k: int = 100,
    virality_k: int = 20,           # top-K threshold for virality
    virality_horizon: int = 4,      # look-ahead steps for virality
) -> dict:
    T = pred_features.shape[0]
    results: dict = {}

    # --- MSE / MAE on log_attention ---
    pred_la = pred_features[:, :, log_attn_idx]   # (T, N)
    true_la = true_features[:, :, log_attn_idx]
    results["mse_log_attn"] = float(np.mean((pred_la - true_la) ** 2))
    results["mae_log_attn"] = float(np.mean(np.abs(pred_la - true_la)))

    # --- MSE / MAE on all features ---
    results["mse_all"] = float(np.mean((pred_features - true_features) ** 2))
    results["mae_all"] = float(np.mean(np.abs(pred_features - true_features)))

    # --- Spearman ρ on log_attention rankings ---
    rhos = []
    for t in range(T):
        if true_la[t].std() < 1e-6:
            continue
        rho, _ = spearmanr(pred_la[t], true_la[t])
        if not np.isnan(rho):
            rhos.append(rho)
    results["spearman_rho"] = float(np.mean(rhos)) if rhos else float("nan")

    # --- Precision@100 (presence) ---
    precisions = []
    for t in range(T):
        pred_set = set(pred_ticker_ids[t].tolist())
        true_set = set(true_ticker_ids[t].tolist()) - {0}   # exclude padding
        if len(true_set) == 0:
            continue
        overlap = len(pred_set & true_set)
        precisions.append(overlap / top_k)
    results["precision_at_100"] = float(np.mean(precisions)) if precisions else float("nan")

    # --- AUC-ROC virality ---
    # virality label: does ticker enter top-K within virality_horizon steps?
    try:
        V = pred_presence.shape[1]
        all_scores, all_labels = [], []
        for t in range(T - virality_horizon):
            # label: in top-K at any step in [t+1, t+virality_horizon]
            future = true_presence[t + 1 : t + virality_horizon + 1]   # (H, V)
            future_topk_mask = np.zeros(V, dtype=float)
            for step_pres in future:
                top_k_idx = np.argsort(step_pres)[-virality_k:]
                future_topk_mask[top_k_idx] = 1.0

            all_scores.append(pred_presence[t])
            all_labels.append(future_topk_mask)

        if all_scores:
            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            if labels.sum() > 0 and labels.sum() < len(labels):
                results["auc_roc_virality"] = float(roc_auc_score(labels, scores))
            else:
                results["auc_roc_virality"] = float("nan")
    except Exception:
        results["auc_roc_virality"] = float("nan")

    return results


def print_metrics(metrics: dict, prefix: str = "") -> None:
    for k, v in metrics.items():
        print(f"  {prefix}{k}: {v:.4f}")
