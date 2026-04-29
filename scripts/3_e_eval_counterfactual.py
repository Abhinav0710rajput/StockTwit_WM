"""
Step 3e: Counterfactual probing

For a set of target tickers (e.g., GME, AMC during the squeeze),
perturb the latent z_t in the direction that maximises the target ticker's
log_attention and measure Δ features for a panel of related/unrelated tickers.

Tests the finite-attention hypothesis:
  - Meme-stock spike → crowd-out effect on unrelated tickers
  - Related tickers (same sector/narrative) → positive contagion

Usage:
    python scripts/3_e_eval_counterfactual.py \
        --model_dir   outputs/rssm_base \
        --data_dir    data/processed \
        --out_dir     outputs/eval/counterfactual \
        --target      GME \
        --eval_tickers GME AMC BB NOK TSLA AAPL MSFT SPY \
        --delta       3.0 \
        --week        2021-01-22
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
from eval.counterfactual import run_counterfactual, plot_counterfactual
from eval.residual_correlation import (
    compute_residuals,
    compute_residual_correlation,
    mean_abs_offdiagonal,
    plot_residual_heatmap,
    save_correlation_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Predefined interesting experiments
EXPERIMENTS = {
    "gme_squeeze": {
        "target": "GME",
        "week": "2021-01-22",
        "delta": 3.0,
        "eval_tickers": ["GME", "AMC", "BB", "NOK", "TSLA", "AAPL", "MSFT", "SPY", "AMZN", "NFLX"],
        "description": "GME squeeze: meme-stock crowd-out experiment",
    },
    "covid_crash": {
        "target": "SPY",
        "week": "2020-02-20",
        "delta": -3.0,
        "eval_tickers": ["SPY", "QQQ", "XLF", "XLE", "GLD", "TLT", "VIX", "AAPL", "AMZN", "TSLA"],
        "description": "COVID crash: broad market shock propagation",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",   type=str, default="outputs/rssm_base")
    p.add_argument("--data_dir",    type=str, default="data/processed")
    p.add_argument("--out_dir",     type=str, default="outputs/eval/counterfactual")
    p.add_argument("--target",      type=str, default=None,
                   help="Target ticker to perturb (use --experiment for presets)")
    p.add_argument("--eval_tickers", nargs="+", default=None)
    p.add_argument("--delta",       type=float, default=3.0,
                   help="Perturbation magnitude on log_attention scale")
    p.add_argument("--week",        type=str, default=None,
                   help="Reference week (YYYY-MM-DD) for counterfactual")
    p.add_argument("--experiment",  type=str, default=None,
                   choices=list(EXPERIMENTS.keys()),
                   help="Run a predefined experiment instead of manual args")
    p.add_argument("--context_len", type=int, default=52)
    p.add_argument("--run_all_experiments", action="store_true",
                   help="Run all predefined experiments sequentially")
    p.add_argument("--run_residual_corr", action="store_true",
                   help="Also run BCE residual correlation diagnostic")
    return p.parse_args()


def run_experiment(
    exp: dict,
    model: TwitWave,
    vocab: Vocabulary,
    panel_train: pd.DataFrame,
    all_panels: dict[str, pd.DataFrame],
    predictor,
    context_len: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    target       = exp["target"]
    week         = exp["week"]
    delta        = exp["delta"]
    eval_tickers = exp["eval_tickers"]
    desc         = exp.get("description", "")

    # Filter eval_tickers to those in vocab
    eval_tickers = [t for t in eval_tickers if vocab.has(t)]
    if not eval_tickers:
        log.warning("None of the eval_tickers are in vocab, skipping %s", desc)
        return
    if not vocab.has(target):
        log.warning("Target ticker %s not in vocab, skipping %s", target, desc)
        return

    log.info("=== %s ===", desc)

    # Build context up to the reference week
    ctx_panel = pd.concat(list(all_panels.values()) + [panel_train])
    ctx_panel = ctx_panel[ctx_panel["week"] <= week].sort_values("week")

    try:
        feat_seq, ids_seq = predictor.build_context(ctx_panel, vocab, max_len=context_len)
    except Exception as e:
        log.error("Context build failed: %s", e)
        return

    df = run_counterfactual(
        model=model,
        vocab=vocab,
        features_seq=feat_seq,
        ticker_ids_seq=ids_seq,
        target_ticker=target,
        delta_log_attn=delta,
        eval_tickers=eval_tickers,
        device=device,
    )

    exp_name = f"{target}_{week}".replace("-", "")
    df.to_csv(out_dir / f"counterfactual_{exp_name}.csv", index=False)

    for feat in ["log_attn", "bullish_rate"]:
        if feat in df["feat"].values:
            fig = plot_counterfactual(
                df=df,
                target_ticker=target,
                feat=feat,
                output_path=out_dir / f"counterfactual_{exp_name}_{feat}.png",
                show=False,
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Summary: top 3 positively / negatively affected tickers
    log_attn_df = df[df["feat"] == "log_attn"].sort_values("delta", ascending=False)
    log.info("Top 3 crowd-in:  %s", log_attn_df.head(3)[["ticker","delta"]].to_string(index=False))
    log.info("Top 3 crowd-out: %s", log_attn_df.tail(3)[["ticker","delta"]].to_string(index=False))

    with open(out_dir / f"counterfactual_{exp_name}_summary.json", "w") as f:
        json.dump({
            "experiment":  desc,
            "target":      target,
            "week":        week,
            "delta":       delta,
            "crowd_in":    log_attn_df.head(3)[["ticker","delta"]].to_dict("records"),
            "crowd_out":   log_attn_df.tail(3)[["ticker","delta"]].to_dict("records"),
        }, f, indent=2)


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

    all_panels = {}
    for split in ["val", "test1", "test2"]:
        p = data_dir / f"panel_{split}.parquet"
        if p.exists():
            all_panels[split] = pd.read_parquet(p)

    # ── Run experiments ───────────────────────────────────────────────────────
    if args.run_all_experiments:
        for exp_name, exp in EXPERIMENTS.items():
            run_experiment(exp, model, vocab, panel_train, all_panels,
                           predictor, args.context_len, out_dir, device)

    elif args.experiment:
        exp = EXPERIMENTS[args.experiment]
        run_experiment(exp, model, vocab, panel_train, all_panels,
                       predictor, args.context_len, out_dir, device)

    elif args.target and args.week:
        eval_tickers = args.eval_tickers or [args.target]
        exp = {
            "target": args.target,
            "week": args.week,
            "delta": args.delta,
            "eval_tickers": eval_tickers,
            "description": f"Custom: perturb {args.target} at {args.week}",
        }
        run_experiment(exp, model, vocab, panel_train, all_panels,
                       predictor, args.context_len, out_dir, device)

    else:
        # Default: run all predefined experiments
        log.info("No specific experiment specified — running all presets")
        for exp_name, exp in EXPERIMENTS.items():
            run_experiment(exp, model, vocab, panel_train, all_panels,
                           predictor, args.context_len, out_dir, device)

    # ── Residual correlation diagnostic ───────────────────────────────────────
    if args.run_residual_corr:
        log.info("=== BCE residual correlation diagnostic ===")
        from eval.predict import Predictor as Pred
        for split in ["test1", "test2"]:
            if split not in all_panels:
                continue
            panel = all_panels[split]
            log.info("Computing residuals for %s...", split)

            presence_logits_list, presence_true_list = [], []
            weeks = sorted(panel["week"].unique())
            for i, week in enumerate(weeks):
                ctx_weeks = sorted(panel_train["week"].unique())[-args.context_len:] + weeks[:i]
                ctx_panel = pd.concat([panel_train, panel])[
                    pd.concat([panel_train, panel])["week"].isin(ctx_weeks)
                ].sort_values("week")
                try:
                    feat_seq, ids_seq = predictor.build_context(ctx_panel, vocab, max_len=args.context_len)
                    logits, true_pres = predictor.get_presence_logits(feat_seq, ids_seq, panel, week, vocab)
                    presence_logits_list.append(logits)
                    presence_true_list.append(true_pres)
                except Exception as e:
                    log.debug("Residual computation failed at %s: %s", week, e)

            if not presence_logits_list:
                continue

            logits_mat = np.stack(presence_logits_list)   # (T, vocab_size)
            true_mat   = np.stack(presence_true_list)     # (T, vocab_size)
            residuals  = compute_residuals(logits_mat, true_mat)

            active = np.where(true_mat.sum(axis=0) > 0)[0].tolist()
            corr   = compute_residual_correlation(residuals, active, top_n=50)
            mad    = mean_abs_offdiagonal(corr)
            log.info("[%s] Mean abs off-diagonal residual correlation: %.4f", split, mad)

            active_tickers = [vocab.decode(idx) for idx in active[:50]]
            save_correlation_csv(corr, active_tickers, out_dir / f"residual_corr_{split}.csv")
            fig = plot_residual_heatmap(
                corr,
                ticker_labels=active_tickers,
                output_path=out_dir / f"residual_heatmap_{split}.png",
                show=False,
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

            with open(out_dir / f"residual_stats_{split}.json", "w") as f:
                json.dump({"mean_abs_offdiag_corr": mad, "n_active": len(active)}, f, indent=2)

    log.info("Counterfactual analysis complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
