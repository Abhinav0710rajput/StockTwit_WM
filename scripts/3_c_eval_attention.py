"""
Step 3c: Cross-ticker attention analysis

Extracts the cross-ticker attention matrix A_t from the set encoder for each
week, computes the diagonal vs off-diagonal ratio (intrinsic vs extrinsic
coupling), and plots attention heatmaps for selected time slices.

Key claim: during meme-stock / contagion events the off-diagonal mass should
increase, meaning the model detects that tickers are more co-dependent.

Usage:
    python scripts/3_c_eval_attention.py \
        --model_dir outputs/rssm_base \
        --data_dir  data/processed \
        --out_dir   outputs/eval/attention \
        --top_n     50
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
from eval.attention_analysis import (
    extract_attention_matrices,
    diagonal_vs_offdiagonal,
    plot_attention_heatmap,
    plot_attention_evolution,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KNOWN_EVENTS = {
    "2020-02-20": "COVID crash",
    "2021-01-22": "GME squeeze",
    "2018-02-05": "Vol spike (XIV)",
    "2020-03-23": "COVID bottom",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="outputs/rssm_base")
    p.add_argument("--data_dir",  type=str, default="data/processed")
    p.add_argument("--out_dir",   type=str, default="outputs/eval/attention")
    p.add_argument("--splits",    nargs="+", default=["test1", "test2"])
    p.add_argument("--top_n",     type=int, default=50,
                   help="Top-N tickers to include in heatmaps")
    p.add_argument("--heatmap_weeks", nargs="+", default=None,
                   help="Specific weeks to plot heatmaps (default: auto-select event weeks)")
    p.add_argument("--context_len", type=int, default=52)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heatmaps").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    vocab = Vocabulary.load(data_dir / "vocab.json")
    panel_train = pd.read_parquet(data_dir / "panel_train.parquet")

    with open(Path(args.model_dir) / "norm_stats.json") as f:
        norm_stats = {k: np.array(v) for k, v in json.load(f).items()}

    model = load_rssm(Path(args.model_dir), len(vocab), device)
    log.info("Model loaded from %s", args.model_dir)

    from eval.predict import Predictor
    predictor = Predictor(model=model, device=device, norm_stats=norm_stats)

    for split in args.splits:
        panel_path = data_dir / f"panel_{split}.parquet"
        if not panel_path.exists():
            log.warning("%s not found, skipping", panel_path)
            continue
        panel = pd.read_parquet(panel_path)
        weeks = sorted(panel["week"].unique())
        log.info("=== %s: extracting attention matrices for %d weeks ===", split, len(weeks))

        attn_by_week: dict[str, np.ndarray] = {}
        ticker_by_week: dict[str, list[str]] = {}

        for i, week in enumerate(weeks):
            ctx_weeks_all = sorted(panel_train["week"].unique())[-args.context_len:] + weeks[:i]
            ctx_panel = pd.concat([panel_train, panel])[
                pd.concat([panel_train, panel])["week"].isin(ctx_weeks_all[-args.context_len:])
            ].sort_values("week")

            try:
                feat_seq, ids_seq = predictor.build_context(ctx_panel, vocab, max_len=args.context_len)
                A_t = predictor.extract_attention(feat_seq, ids_seq)  # (K, K)
                if A_t is not None:
                    attn_by_week[str(week)] = A_t.cpu().numpy()
                    active = ids_seq[-1].cpu().numpy()
                    ticker_by_week[str(week)] = [
                        vocab.decode(int(idx)) for idx in active if int(idx) != 0
                    ]
            except Exception as e:
                log.debug("Attention extraction failed at %s: %s", week, e)
                continue

        log.info("Extracted attention for %d/%d weeks", len(attn_by_week), len(weeks))

        # ── Diagonal vs off-diagonal time series ──────────────────────────────
        if attn_by_week:
            week_list = sorted(attn_by_week.keys())
            diag_ratio = []
            for w in week_list:
                A = attn_by_week[w]
                d, od = diagonal_vs_offdiagonal(A)
                diag_ratio.append({"week": w, "diag_mean": d, "offdiag_mean": od,
                                   "coupling_ratio": od / (d + 1e-8)})
            coupling_df = pd.DataFrame(diag_ratio)
            coupling_df.to_csv(out_dir / f"coupling_{split}.csv", index=False)

            fig = plot_attention_evolution(
                coupling_df,
                known_events=KNOWN_EVENTS,
                output_path=out_dir / f"coupling_{split}.pdf",
                show=False,
            )
            fig.savefig(out_dir / f"coupling_{split}.png", dpi=150, bbox_inches="tight")

        # ── Heatmaps for selected weeks ────────────────────────────────────────
        heatmap_weeks = args.heatmap_weeks
        if heatmap_weeks is None:
            # Auto: pick event weeks + 1 week before/after
            heatmap_weeks = []
            for event_week in KNOWN_EVENTS:
                for w in attn_by_week:
                    if abs((pd.Timestamp(w) - pd.Timestamp(event_week)).days) < 14:
                        heatmap_weeks.append(w)

        for w in heatmap_weeks:
            if w not in attn_by_week:
                continue
            A = attn_by_week[w][:args.top_n, :args.top_n]
            labels = ticker_by_week.get(w, [])[:args.top_n]
            event = KNOWN_EVENTS.get(w, w)
            fig = plot_attention_heatmap(
                A,
                ticker_labels=labels,
                title=f"Cross-ticker attention: {w} ({event})",
                output_path=out_dir / "heatmaps" / f"attn_{split}_{w}.png",
                show=False,
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

        log.info("[%s] Plots saved to %s", split, out_dir)

    log.info("Attention analysis complete.")


if __name__ == "__main__":
    main()
